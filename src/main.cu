#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "constants.cuh"
#include "renderer.cuh"
#include "scene.cuh"
#include "sphere.cuh"

#include <chrono>
#include <iostream>
#include <vector>

int main() {
	unsigned char* image = nullptr;
	cudaMalloc(&image, IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL);

	std::vector<Sphere> h_spheres;
	h_spheres.push_back(Sphere({ 0.0f, -100.5f, -1.0f}, 100.0f, Material(make_float3(0.0, 0.8, 0.7), 0.0f, 0.0f, 0)));
	h_spheres.push_back(Sphere({ 0.0f,    0.5f, -1.0f},   0.5f, Material(make_float3(0.7, 0.3, 0.9), 0.0f, 0.0f, 0)));
	h_spheres.push_back(Sphere({-0.9f,    0.0f, -1.0f},   0.5f, Material(make_float3(0.8, 0.5, 0.5), 0.1f, 0.0f, 1)));
	h_spheres.push_back(Sphere({ 0.9f,    0.0f, -1.0f},   0.5f, Material(make_float3(0.8, 0.6, 0.2), 0.5f, 0.0f, 1)));
	h_spheres.push_back(Sphere({ 0.0f,   -0.3f, -1.0f},   0.2f, Material(make_float3(0.8, 0.8, 0.8), 0.0f, 0.0f, 1)));
	int numSpheres = h_spheres.size();

	Sphere* d_spheres;
	cudaMalloc(&d_spheres, numSpheres * sizeof(Sphere));
	cudaMemcpy(d_spheres, h_spheres.data(), numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);

	Camera h_camera(
		make_float3(0.0f, 0.0f, 0.0f), // lookFrom
		make_float3(0.0f, 0.0f, -1.0f), // lookAt
		make_float3(0.0f, 1.0f, 0.0f), // vUp
		90.0f, // vFov
		5.0f, // focusDistance
		0.0f  // aperture
	);

	Scene h_scene(d_spheres, numSpheres, h_camera);

	Scene* d_scene;
	cudaMalloc(&d_scene, sizeof(Scene));
	cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);

	printf("Running kernel...\n");
	std::chrono::steady_clock::time_point beginRender = std::chrono::steady_clock::now();

	renderImage(image, d_scene);

	unsigned char* hostImage = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL];
	cudaMemcpy(hostImage, image, IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL, cudaMemcpyDeviceToHost);
	
	std::chrono::steady_clock::time_point endRender = std::chrono::steady_clock::now();
	printf("Done in %f s \n", (float)std::chrono::duration_cast<std::chrono::microseconds>(endRender - beginRender).count() / 1000000);

	printf("Parsing results...\n");
	std::chrono::steady_clock::time_point beginParse = std::chrono::steady_clock::now();

	if (stbi_write_png("output.png", IMAGE_WIDTH, IMAGE_HEIGHT, BYTES_PER_PIXEL, hostImage, IMAGE_WIDTH * BYTES_PER_PIXEL)) {
		std::cout << "Saved output.png" << std::endl;
	} else {
		std::cerr << "Failed to write PNG" << std::endl;
	}

	std::chrono::steady_clock::time_point endParse = std::chrono::steady_clock::now();
	printf("Done in %f s \n", (float)std::chrono::duration_cast<std::chrono::microseconds>(endParse - beginParse).count() / 1000000);

	cudaFree(image);
	cudaFree(d_spheres);
	cudaFree(d_scene);
	delete[] hostImage;
	return 0;
}