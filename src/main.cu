#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "json_utils.cuh"
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

	HostSceneData scene;
	try {
		scene = loadSceneDescription("assets/scenes/demo_scene.json");
	} catch (const std::exception& e) {
		printf("Scene load failed: %s\n", e.what());
		return 1;
	}

	printf(" === Chosen device === \n");
	printf("Number: %d\n", device);
	printf("Name: %s\n", prop.name);
	printf("Compute capability: %d.%d\n", prop.major, prop.minor);
	printf("Total global memory: %llu MB\n", (unsigned long long)(prop.totalGlobalMem / (1024 * 1024)));

	printf("=== Parsing scene data === \n");
	std::chrono::steady_clock::time_point beginParse = std::chrono::steady_clock::now();

	unsigned char* image = nullptr;
	cudaMalloc(&image, scene.imageWidth * scene.imageHeight * 3);

	CudaBuffer<Material> materials;
	materials.allocate(scene.materials.size());
	materials.copyFromHost(scene.materials.data());

	Sphere* d_spheres = nullptr;
	int numSpheres = scene.spheres.size();
	cudaMalloc(&d_spheres, numSpheres * sizeof(Sphere));
	cudaMemcpy(d_spheres, scene.spheres.data(),
			numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);


	Triangle* h_triangles = nullptr;
	int numTriangles = 0;
	BoundingBox* h_boundingBox = nullptr;
	
	std::ifstream meshFile(scene.mesh.obj_path);
	if (!meshFile) {
		printf("Unable to open OBJ file\n");
		return 1;
	}

	loadTriangles(
		meshFile,
		scene.mesh.scale,
		scene.mesh.translation,
		scene.mesh.rotation,
		scene.mesh.materialId,
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

	Camera h_camera = scene.camera;

	Scene h_scene(
		materials.ptr, materials.size,
		d_spheres, numSpheres,
		d_triangles, numTriangles,
		d_boundingBox,
		d_nodes, h_nodes.size(),
		d_triIndices,
		texObj, width, height,
		h_camera,
		scene.imageWidth, scene.imageHeight,
		scene.samplesPerPixel, scene.maxBounces
	);

	Scene* d_scene;
	cudaMalloc(&d_scene, sizeof(Scene));
	cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);

	std::chrono::steady_clock::time_point endParse = std::chrono::steady_clock::now();
	printf("Done in %f s \n", static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(endParse - beginParse).count()) / 1000000);

	printf("=== Running kernel === \n");
	std::chrono::steady_clock::time_point beginRender = std::chrono::steady_clock::now();
	renderImage(image, d_scene, scene.imageWidth, scene.imageHeight);

	unsigned char* hostImage = new unsigned char[scene.imageWidth * scene.imageHeight * 3];
	cudaMemcpy(hostImage, image, scene.imageWidth * scene.imageHeight * 3, cudaMemcpyDeviceToHost);
	
	std::chrono::steady_clock::time_point endRender = std::chrono::steady_clock::now();
	printf("Done in %f s \n", static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(endRender - beginRender).count()) / 1000000);

	printf(" === Writing image === \n");
	std::chrono::steady_clock::time_point beginWriting = std::chrono::steady_clock::now();

	if (!stbi_write_png("output.png", scene.imageWidth, scene.imageHeight, 3, hostImage, scene.imageWidth * 3))
		printf("Failed to write PNG");

	std::chrono::steady_clock::time_point endWriting = std::chrono::steady_clock::now();
	printf("Done in %f s \n", static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(endWriting - beginWriting).count()) / 1000000);

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