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

	std::vector<Sphere> h_spheres;
	h_spheres.push_back(Sphere({ 0.0f, -100.5f, -1.0f}, 100.0f, Material(make_float3(0.0f, 0.8f, 0.7f), 0.0f, 0.0f, 0)));
	h_spheres.push_back(Sphere({ -4.5f,   1.8f, 0.5f},   2.5f, Material(make_float3(0.8f, 0.8f, 0.2f), 0.1f, 0.0f, 1)));
	h_spheres.push_back(Sphere({ 0.0f,   1.8f, -4.5f},   2.5f, Material(make_float3(0.2f, 0.8f, 0.4f), 0.2f, 0.0f, 1)));
	h_spheres.push_back(Sphere({ 2.1f,  0.27f, 2.0f},   0.8f, Material(make_float3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f, 2)));
	int numSpheres = h_spheres.size();

	std::ifstream meshFile("assets/utah_teapot.obj");
	if (!meshFile) {
		printf("Unable to open OBJ file");
		return 1;
	}

	Triangle* h_triangles = nullptr;
	int numTriangles = 0;
	BoundingBox* h_boundingBox = nullptr;

	loadTriangles(
		meshFile,
		make_float3(1.0f, 1.0f, 1.0f), // scaling
		make_float3(0.0f, -0.6f, 0.0f), // translation
		make_float3(0.0f, -45.0f, 0.0f), // rotation (degrees)
		Material(make_float3(0.7, 0.3, 0.9), 0.7f, 0.0f, 1),
		h_triangles,
		numTriangles,
		h_boundingBox
	);
	meshFile.close();

	Triangle* d_triangles;
	cudaMalloc(&d_triangles, numTriangles * sizeof(Triangle));
	cudaMemcpy(d_triangles, h_triangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

	BoundingBox* d_boundingBox;
	cudaMalloc(&d_boundingBox, sizeof(BoundingBox));
	cudaMemcpy(d_boundingBox, h_boundingBox, sizeof(BoundingBox), cudaMemcpyHostToDevice);

	Sphere* d_spheres;
	cudaMalloc(&d_spheres, numSpheres * sizeof(Sphere));
	cudaMemcpy(d_spheres, h_spheres.data(), numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);

	Camera h_camera(
		make_float3(6.0f, 2.5f, 3.0f), // lookFrom
		make_float3(0.0f, 1.0f, 0.0f), // lookAt
		make_float3(0.0f, 1.0f, 0.0f), // vUp
		60.0f, // vFov
		5.0f, // focusDistance
		0.0f  // aperture
	);

	Scene h_scene(d_spheres, numSpheres, d_triangles, numTriangles, h_boundingBox, h_camera);

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
	delete[] hostImage;
	return 0;
}