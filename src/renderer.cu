#include "renderer.cuh"

#include "../lib/stb_image_write.h"

#include "constants.cuh"

__device__ uchar4 shadePixel(int x, int y) {
	uchar4 p;
	p.x = static_cast<unsigned char>(255.0f * x / IMAGE_WIDTH);
	p.y = static_cast<unsigned char>(255.0f * y / IMAGE_HEIGHT);
	p.z = 0;
	p.w = 255;
	return p;
}

__global__ void renderKernel(unsigned char* image, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;

	int idx = y * width + x;
	uchar4 pixel = shadePixel(x, y);

	image[idx * 3 + 0] = pixel.x;
	image[idx * 3 + 1] = pixel.y;
	image[idx * 3 + 2] = pixel.z;
}

void renderImage(unsigned char* image, int width, int height) {
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x,
			  (height + block.y - 1) / block.y);

	renderKernel<<<grid, block>>>(image, width, height);
	cudaDeviceSynchronize();
}
