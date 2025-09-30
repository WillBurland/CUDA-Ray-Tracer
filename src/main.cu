#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "mesh_loader.cuh"
#include "renderer.cuh"

#include <chrono>

int main() {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		printf("No CUDA devices found.\n");
		return 1;
	}

	printf(" === Device list === \n");
	for (int i = 0; i < deviceCount; ++i) {
		cudaDeviceProp p;
		cudaGetDeviceProperties(&p, i);
		printf("Device %d: %s (compute %d.%d)\n", i, p.name, p.major, p.minor);
	}

	int device;
	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	printf(" === Chosen device === \n");
	printf("Number: %d\n", device);
	printf("Name: %s\n", prop.name);
	printf("Compute capability: %d.%d\n", prop.major, prop.minor);
	printf("Total global memory: %llu MB\n", (unsigned long long)(prop.totalGlobalMem / (1024 * 1024)));

	if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) {
		printf("GPU compute capability %d.%d is too old. Requires >=7.5.\n", prop.major, prop.minor);
		return 1;
	}

	printf("=== Parsing scene data === \n");
	std::chrono::steady_clock::time_point beginParse = std::chrono::steady_clock::now();

	unsigned char* image = nullptr;
	cudaMalloc(&image, IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL);

	std::vector<Material> h_materials;
	h_materials.push_back(Material::Lambertian(0, make_float3(0.3f, 0.5f, 0.4f)));
	h_materials.push_back(Material::Metal(1, make_float3(0.8f, 0.8f, 0.3f), 0.0f));
	h_materials.push_back(Material::Metal(2, make_float3(0.3f, 0.8f, 0.3f), 0.05f));
	h_materials.push_back(Material::Metal(3, make_float3(0.3f, 0.3f, 0.8f), 0.2f));
	h_materials.push_back(Material::Transparent(4, 1.5f));
	h_materials.push_back(Material::Emissive(5, make_float3(1.0f, 0.4f, 0.2f), 4.0f));
	int numMaterials = h_materials.size();

	std::vector<Sphere> h_spheres;
	h_spheres.push_back(Sphere({  0.0f, -100.5f, -1.0f}, 100.0f, 0));
	h_spheres.push_back(Sphere({ -4.5f,    1.8f,  0.5f},   2.5f, 1));
	h_spheres.push_back(Sphere({  0.0f,    1.8f, -4.5f},   2.5f, 2));
	h_spheres.push_back(Sphere({  2.1f,    0.27f, 2.0f},   0.8f, 4));
	h_spheres.push_back(Sphere({  2.5f,   -0.2f,  0.8f},   0.4f, 5));
	int numSpheres = h_spheres.size();

	std::ifstream meshFile("assets/models/utah_teapot.obj");
	if (!meshFile) {
		printf("Unable to open OBJ file");
		return 1;
	}

	Triangle* h_triangles = nullptr;
	int numTriangles = 0;
	BoundingBox* h_boundingBox = nullptr;

	loadTriangles(
		meshFile,
		make_float3(1.0f,   1.0f, 1.0f), // scaling
		make_float3(0.0f,  -0.6f, 0.0f), // translation
		make_float3(0.0f, -45.0f, 0.0f), // rotation (degrees)
		3,
		h_triangles,
		numTriangles,
		h_boundingBox,
		true
	);
	meshFile.close();

	std::vector<int> triIndices(numTriangles);
	for (int i = 0; i < numTriangles; i++)
		triIndices[i] = i;

	std::vector<BVHNode> h_nodes(numTriangles * 2);
	int nodeCount = 0;
	std::vector<Triangle> h_triVec(h_triangles, h_triangles + numTriangles);

	buildBVH(h_nodes.data(), nodeCount, triIndices, 0, numTriangles, h_triVec);

	h_nodes.resize(nodeCount);

	BVHNode* d_nodes;
	cudaMalloc(&d_nodes, h_nodes.size() * sizeof(BVHNode));
	cudaMemcpy(d_nodes, h_nodes.data(), h_nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

	int* d_triIndices;
	cudaMalloc(&d_triIndices, triIndices.size() * sizeof(int));
	cudaMemcpy(d_triIndices, triIndices.data(), triIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

	int width, height, channels;
	float* data = stbi_loadf("assets/images/skybox.hdr", &width, &height, &channels, 3);
	if (!data) {
		printf("Unable to open image file");
		return 1;
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);

	std::vector<float4> h_pixels(width * height);
	for (int i = 0; i < width * height; i++) {
		h_pixels[i] = make_float4(
			data[i * 3 + 0],
			data[i * 3 + 1],
			data[i * 3 + 2],
			1.0f
		);
	}

	cudaMemcpy2DToArray(
		cuArray,
		0,
		0,
		h_pixels.data(),
		width * sizeof(float4),
		width * sizeof(float4),
		height,
		cudaMemcpyHostToDevice
	);

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc = {};
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

	Triangle* d_triangles;
	cudaMalloc(&d_triangles, numTriangles * sizeof(Triangle));
	cudaMemcpy(d_triangles, h_triangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

	BoundingBox* d_boundingBox;
	cudaMalloc(&d_boundingBox, sizeof(BoundingBox));
	cudaMemcpy(d_boundingBox, h_boundingBox, sizeof(BoundingBox), cudaMemcpyHostToDevice);

	Sphere* d_spheres;
	cudaMalloc(&d_spheres, numSpheres * sizeof(Sphere));
	cudaMemcpy(d_spheres, h_spheres.data(), numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);

	Material* d_materials;
	cudaMalloc(&d_materials, numMaterials * sizeof(Material));
	cudaMemcpy(d_materials, h_materials.data(), numMaterials * sizeof(Material), cudaMemcpyHostToDevice);

	Camera h_camera(
		make_float3(6.0f, 2.5f, 3.0f), // lookFrom
		make_float3(0.0f, 1.0f, 0.0f), // lookAt
		make_float3(0.0f, 1.0f, 0.0f), // vUp
		60.0f, // vFov
		5.0f, // focusDistance
		50.0f  // f-stop
	);

	Scene h_scene(
		d_materials, numMaterials,
		d_spheres, numSpheres,
		d_triangles, numTriangles,
		d_boundingBox,
		d_nodes, h_nodes.size(),
		d_triIndices,
		texObj, width, height,
		h_camera
	);

	Scene* d_scene;
	cudaMalloc(&d_scene, sizeof(Scene));
	cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);

	std::chrono::steady_clock::time_point endParse = std::chrono::steady_clock::now();
	printf("Done in %f s \n", (float)std::chrono::duration_cast<std::chrono::microseconds>(endParse - beginParse).count() / 1000000);

	printf("=== Running kernel === \n");
	std::chrono::steady_clock::time_point beginRender = std::chrono::steady_clock::now();

	renderImage(image, d_scene);

	unsigned char* hostImage = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL];
	cudaMemcpy(hostImage, image, IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL, cudaMemcpyDeviceToHost);
	
	std::chrono::steady_clock::time_point endRender = std::chrono::steady_clock::now();
	printf("Done in %f s \n", (float)std::chrono::duration_cast<std::chrono::microseconds>(endRender - beginRender).count() / 1000000);

	printf(" === Writing image === \n");
	std::chrono::steady_clock::time_point beginWriting = std::chrono::steady_clock::now();

	if (!stbi_write_png("output.png", IMAGE_WIDTH, IMAGE_HEIGHT, BYTES_PER_PIXEL, hostImage, IMAGE_WIDTH * BYTES_PER_PIXEL))
		printf("Failed to write PNG");

	std::chrono::steady_clock::time_point endWriting = std::chrono::steady_clock::now();
	printf("Done in %f s \n", (float)std::chrono::duration_cast<std::chrono::microseconds>(endWriting - beginWriting).count() / 1000000);

	cudaFree(image);
	cudaFree(d_spheres);
	cudaFree(d_triangles);
	cudaFree(d_boundingBox);
	cudaFree(d_scene);
	free(h_triangles);
	free(h_boundingBox);
	stbi_image_free(data);
	delete[] hostImage;
	return 0;
}