#pragma once

struct Ray {
	float3 origin;
	float3 direction;
	float3 invDirection;

	__device__ Ray(float3 origin, float3 direction, float3 invDirection) :
		origin(origin),
		direction(direction),
		invDirection(invDirection) {}

	__device__ Ray(Camera camera, float u, float v, ulong* seed) {
		if (camera.aperture <= 0) {
			origin = camera.origin;
		} else {
			float3 p = randVecInUnitDisk(seed);
			origin = camera.origin + camera.defocusDiscU * p.x + camera.defocusDiscV * p.y;
		}

		direction = unit(camera.lowerLeftCorner + camera.horizontal * u + camera.vertical * v - origin);
		invDirection = inv(direction);
	}
};

__device__ float3 rayAt(Ray ray, float t) {
	return ray.origin + ray.direction * t;
}