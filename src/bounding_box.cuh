#pragma once

#include "vec_utils.cuh"

struct BoundingBox {
	int id, parentId;
	float3 min, max;

	__host__ BoundingBox(int id, int parentId, float3 min, float3 max) :
		id(id),
		parentId(parentId),
		min(min),
		max(max) {}
};