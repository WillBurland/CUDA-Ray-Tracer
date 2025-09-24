#pragma once

struct BoundingBox {
	int id, parentId;
	float3 min, max;

	BoundingBox() :
		id(-1),
		parentId(-1),
		min(make_float3(0.0f, 0.0f, 0.0f)),
		max(make_float3(0.0f, 0.0f, 0.0f)) {}

	BoundingBox(int id, int parentId, float3 min, float3 max) :
		id(id),
		parentId(parentId),
		min(min),
		max(max) {}
};