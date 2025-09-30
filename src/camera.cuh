#pragma once

#include "constants.cuh"

struct Camera {
	float3 origin;
	float3 horizontal, vertical, lowerLeftCorner;
	float3 defocusDiscU, defocusDiscV;
	float focusDistance, aperture;

	__host__ Camera(const float3 lookFrom, const float3 lookAt, const float3 vUp, const float vFov, const float focusDistance, const float fStop) {
		const float viewportHeight = 2.0f * tanf(vFov * 3.141592654f / 360.0f) * focusDistance;

		const float3 w = unit(lookFrom - lookAt);
		const float3 u = unit(cross(vUp, w));
		const float3 v = cross(w, u);

		const float3 viewportU = u * ASPECT_RATIO * viewportHeight;
		const float3 viewportV = v * viewportHeight;

		const float aperture = 2.0f * atanf(0.5f / fStop);
		const float defocusRadius = focusDistance * tanf(aperture * 0.5f);

		this->origin          = lookFrom;
		this->horizontal      = viewportU;
		this->vertical        = viewportV;
		this->lowerLeftCorner = lookFrom - viewportU * 0.5f - viewportV * 0.5f - w * focusDistance;
		this->defocusDiscU    = u * defocusRadius;
		this->defocusDiscV    = v * defocusRadius;
		this->focusDistance   = focusDistance;
		this->aperture        = aperture;
	}
};