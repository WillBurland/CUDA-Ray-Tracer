#pragma once

#include "vec_utils.cuh"

enum MaterialType {
	UNDEFINED   = -1,
	LAMBERTIAN  = 0,
	METAL       = 1,
	TRANSPARENT = 2,
	EMISSIVE    = 3
};

struct Material {
	int id{-1};
	float3 albedo{make_float3(0.0f)};
	float fuzz{0.0f};
	float ior{0.0f};
	int type{UNDEFINED};

	__device__ Material() :
		id(-1),
		albedo(make_float3(0.0f)),
		fuzz(0.0f),
		ior(0.0f),
		type(UNDEFINED) {}

	__host__ Material(const int id, const float3 albedo, const float fuzz, const float ior, const int type) :
		id(id),
		albedo(albedo),
		fuzz(fuzz),
		ior(ior),
		type(type) {}

	__host__ static inline Material Lambertian(const int id, const float3 albedo) {
		return Material(id, albedo, 0.0f, 0.0f, LAMBERTIAN);
	}

	__host__ static inline Material Metal(const int id, const float3 albedo, const float fuzz) {
		return Material(id, albedo, fuzz, 0.0f, METAL);
	}

	__host__ static inline Material Transparent(const int id, const float ior) {
		return Material(id, make_float3(1.0f), 0.0f, ior, TRANSPARENT);
	}

	__host__ static inline Material Emissive(const int id, const float3 albedo, const float power) {
		return Material(id, albedo * power, 0.0f, 0.0f, EMISSIVE);
	}
};