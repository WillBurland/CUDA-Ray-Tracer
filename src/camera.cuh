#pragma once

#include "vec_utils.cuh"

struct Camera {
	float3 origin{make_float3(0.0f)};
	float3 horizontal{make_float3(0.0f)}, vertical{make_float3(0.0f)}, lowerLeftCorner{make_float3(0.0f)};
	float3 defocusDiscU{make_float3(0.0f)}, defocusDiscV{make_float3(0.0f)};
	float focusDistance{0.0f}, aperture{0.0f};

	__host__ Camera(const float3 lookFrom, const float3 lookAt, const float3 vUp,  const float vFov, const float focusDistance, const float fStop, const float aspectRatio) {
		const float viewportHeight = 2.0f * tanf(vFov * 3.141592654f / 360.0f) * focusDistance;

		const float3 w = unit(lookFrom - lookAt);
		const float3 u = unit(cross(vUp, w));
		const float3 v = cross(w, u);

		const float3 viewportU = u * aspectRatio * viewportHeight;
		const float3 viewportV = v * viewportHeight;

		const float apertureRadians = 2.0f * atanf(0.5f / fStop);
		const float defocusRadius = focusDistance * tanf(apertureRadians * 0.5f);

		this->origin          = lookFrom;
		this->horizontal      = viewportU;
		this->vertical        = viewportV;
		this->lowerLeftCorner = lookFrom - viewportU * 0.5f - viewportV * 0.5f - w * focusDistance;
		this->defocusDiscU    = u * defocusRadius;
		this->defocusDiscV    = v * defocusRadius;
		this->focusDistance   = focusDistance;
		this->aperture        = apertureRadians;
	}

	__host__ Camera() {};
};