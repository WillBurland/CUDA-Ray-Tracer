#pragma once

#include "constants.cuh"
#include "vec_utils.cuh"

struct Camera {
	float3 origin;
	float3 horizontal, vertical, lowerLeftCorner;
	float3 defocusDiscU, defocusDiscV;
	float3 blockSize, blockOffset;
	float focusDistance, aperture;

	Camera(float3 lookFrom, float3 lookAt, float3 vUp, float vFov, float focusDistance, float fStop) {
		float aperture = 2.0f * atanf(1.0f / (2.0f * fStop));

		float theta = vFov * (float)3.141592654 / 180.0f;
		float h = tan(theta / 2);
		float viewportHeight = 2.0f * h * focusDistance;
		float viewportWidth  = ASPECT_RATIO * viewportHeight;

		float3 w = unit(lookFrom - lookAt);
		float3 u = unit(cross(vUp, w));
		float3 v = cross(w, u);

		float3 viewportU   = u * viewportWidth;
		float3 viewportV   = v * viewportHeight;
		float3 pixelDeltaU = viewportU / (float)IMAGE_WIDTH;
		float3 pixelDeltaV = viewportV / (float)IMAGE_HEIGHT;
		float3 viewportLowerLeftCorner = ((lookFrom - (viewportU / 2.0f)) - (viewportV / 2.0f)) - (w * focusDistance);
		
		float defocusRadius = focusDistance * tan(aperture / 2.0f);
		float3 defocusDiscU = u * defocusRadius;
		float3 defocusDiscV = v * defocusRadius;

		this->blockOffset     = make_float3(0.0f, 0.0f, 0.0f);
		this->origin          = lookFrom;
		this->horizontal      = viewportU;
		this->vertical        = viewportV;
		this->lowerLeftCorner = viewportLowerLeftCorner;
		this->defocusDiscU    = defocusDiscU;
		this->defocusDiscV    = defocusDiscV;
		this->focusDistance   = focusDistance;
		this->aperture        = aperture;
	}
};