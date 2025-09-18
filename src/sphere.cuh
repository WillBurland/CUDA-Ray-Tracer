#pragma once

#include "material.cuh"

struct Sphere {
	float3 centre;
	float radius;
	Material material;

	Sphere(float3 centre, float radius, Material material) :
		centre(centre),
		radius(radius),
		material(material) {}
};
