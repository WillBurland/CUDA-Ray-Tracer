#pragma once

#include "material.cuh"
#include "ray.cuh"

struct HitRecord {
	float3 p;
	float3 normal;
	float t;
	bool frontFace;
	Material material;

	__device__ HitRecord() :
		p(make_float3(0.0f, 0.0f, 0.0f)),
		normal(make_float3(0.0f, 0.0f, 0.0f)),
		t(0.0f),
		frontFace(false),
		material() {}
};

__device__ void setFaceNormal(HitRecord* hitRecord, Ray ray, float3 outwardNormal) {
	hitRecord->frontFace = dot(ray.direction, outwardNormal) < 0;
	hitRecord->normal = hitRecord->frontFace ? outwardNormal : outwardNormal * -1;
}

__device__ bool hitSphere(Sphere s, Ray r, float tMin, float tMax, HitRecord* hit) {
	float3 oc = r.origin - s.centre;
	float a = length2(r.direction);
	float halfB = dot(oc, r.direction);
	float c = length2(oc) - s.radius * s.radius;
	float discriminant = halfB * halfB - a * c;

	if (discriminant < 0)
		return false;

	float sqrtD = sqrt(discriminant);
	float root = (-halfB - sqrtD) / a;
	if (root < tMin || tMax < root) {
		root = (-halfB + sqrtD) / a;
		if (root < tMin || tMax < root)
			return false;
	}

	hit->t = root;
	hit->p = rayAt(r, root);
	float3 outwardNormal = (hit->p - s.centre) / s.radius;
	setFaceNormal(hit, r, outwardNormal);
	hit->material = s.material;

	return true;
}

__device__ bool hitAnything(HitRecord* hitRecord, Ray ray, float tMin, float tMax, Scene* scene) {
	HitRecord temp;
	bool hitAnything = false;
	float closestSoFar = INFINITY;

	for (int i = 0; i < scene->numSpheres; i++) {
		if (hitSphere(scene->spheres[i], ray, tMin, closestSoFar, &temp)) {
			hitAnything = true;
			closestSoFar = temp.t;
			*hitRecord = temp;
		}
	}

	return hitAnything;
}