#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "constants.cuh"
#include "renderer.cuh"

#include <chrono>
#include <iostream>

int main() {
	unsigned char* image = nullptr;
	cudaMalloc(&image, IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL);

	printf("Running kernel...\n");
	std::chrono::steady_clock::time_point beginRender = std::chrono::steady_clock::now();

	renderImage(image, IMAGE_WIDTH, IMAGE_HEIGHT);

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
	delete[] hostImage;
	return 0;
}