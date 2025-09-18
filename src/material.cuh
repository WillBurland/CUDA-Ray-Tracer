#pragma once

struct Material {
	float3 albedo;
	float fuzz;
	float ior;
	int type;

   __device__ Material() :
		albedo(make_float3(0.0f, 0.0f, 0.0f)),
		fuzz(0.0f),
		ior(1.0f),
		type(0) {}

	Material(float3 albedo, float fuzz, float ior, int type) :
		albedo(albedo),
		fuzz(fuzz),
		ior(ior),
		type(type) {}
};