#pragma once

struct Sphere {
	float3 centre;
	float radius;
	int materialId;

	__host__ Sphere(const float3 centre, const float radius, const int materialId) :
		centre(centre),
		radius(radius),
		materialId(materialId) {}
};
