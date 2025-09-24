#pragma once

#include "vec_utils.cuh"

enum MaterialType {
	LAMBERTIAN = 0,
	METAL      = 1,
	TRANSPARENT = 2,
	EMISSIVE   = 3
};

struct Material {
	float3 albedo;
	float fuzz;
	float ior;
	int type;

   __host__ __device__ Material() :
		albedo(make_float3(0.0f, 0.0f, 0.0f)),
		fuzz(0.0f),
		ior(1.0f),
		type(0) {}

	Material(float3 albedo, float fuzz, float ior, int type) :
		albedo(albedo),
		fuzz(fuzz),
		ior(ior),
		type(type) {}

	static Material Lambertian(float3 albedo) {
		return Material(albedo, 0.0f, 0.0f, LAMBERTIAN);
	}

	static Material Metal(float3 albedo, float fuzz) {
		return Material(albedo, fuzz, 0.0f, METAL);
	}

	static Material Transparent(float ior) {
		return Material(make_float3(1.0f, 1.0f, 1.0f), 0.0f, ior, TRANSPARENT);
	}

	static Material Emissive(float3 albedo, float power) {
		return Material(albedo * power, 0.0f, 0.0f, EMISSIVE);
	}
};