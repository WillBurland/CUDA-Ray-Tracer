#pragma once

#include "vec_utils.cuh"

struct BoundingBox {
	int id, parentId;
	float3 min, max;

	BoundingBox() :
		id(-1),
		parentId(-1),
		min(make_float3(0.0f)),
		max(make_float3(0.0f)) {}

	BoundingBox(int id, int parentId, float3 min, float3 max) :
		id(id),
		parentId(parentId),
		min(min),
		max(max) {}

	__host__ __device__ BoundingBox(float3 min_, float3 max_) 
		: id(-1), parentId(-1), min(min_), max(max_) {}
};