#pragma once

#include "scene.cuh"

__host__ void renderImage(unsigned char* image, const Scene* scene, const int width, const int height);