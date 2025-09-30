#pragma once

struct Ray {
	float3 origin, direction, invDirection;

	__device__ Ray(const float3 origin, const float3 direction, const float3 invDirection) :
		origin(origin),
		direction(unit(direction)),
		invDirection(invDirection) {}

	__device__ Ray(const Camera camera, const float u, const float v, ulong* seed) {
		if (camera.aperture <= 0) {
			this->origin = camera.origin;
		} else {
			const float3 p = randVecInUnitDisk(seed);
			this->origin = camera.origin + camera.defocusDiscU * p.x + camera.defocusDiscV * p.y;
		}

		this->direction = unit(camera.lowerLeftCorner + camera.horizontal * u + camera.vertical * v - this->origin);
		this->invDirection = inv(this->direction);
	}

	__device__ float3 at(const float t) const {
		return this->origin + this->direction * t;
	}
};