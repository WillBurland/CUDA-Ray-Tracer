#pragma once

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
	float halfB = dot(oc, r.direction);
	float c = length2(oc) - s.radius * s.radius;
	float discriminant = halfB * halfB - c;

	if (discriminant < 0)
		return false;

	float sqrtD = sqrtf(discriminant);
	float root = -halfB - sqrtD;
	if (root < tMin || tMax < root) {
		root = -halfB + sqrtD;
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

__device__ bool hitTriangle(Triangle t, Ray r, float tMin, float tMax, HitRecord* hit) {
	float3 edge1 = t.p1 - t.p0;
	float3 edge2 = t.p2 - t.p0;
	float3 h = cross(r.direction, edge2);
	float a = dot(edge1, h);

	if (fabsf(a) < 1e-8f)
		return false;

	float f = 1.0f / a;
	float3 s = r.origin - t.p0;
	float u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f)
		return false;

	float3 q = cross(s, edge1);
	float v = f * dot(r.direction, q);

	if (v < 0.0f || u + v > 1.0f)
		return false;

	float t0 = f * dot(edge2, q);

	if (t0 > tMin && t0 < tMax)	{
		hit->t = t0;
		hit->p = rayAt(r, t0);
		hit->material = t.material;
		setFaceNormal(hit, r, unit(t.smoothNormals ? (1.0f - u - v) * t.n0 + u * t.n1 + v * t.n2 : cross(edge1, edge2)));

		return true;
	}
	
	return false;
}

__device__ bool hitBoundingBox(BoundingBox box, Ray r) {
	float tx1 = (box.min.x - r.origin.x) * r.invDirection.x;
	float tx2 = (box.max.x - r.origin.x) * r.invDirection.x;

	float tmin = fminf(tx1, tx2);
	float tmax = fmaxf(tx1, tx2);

	float ty1 = (box.min.y - r.origin.y) * r.invDirection.y;
	float ty2 = (box.max.y - r.origin.y) * r.invDirection.y;

	tmin = fmaxf(tmin, fminf(ty1, ty2));
	tmax = fminf(tmax, fmaxf(ty1, ty2));

	float tz1 = (box.min.z - r.origin.z) * r.invDirection.z;
	float tz2 = (box.max.z - r.origin.z) * r.invDirection.z;

	tmin = fmaxf(tmin, fminf(tz1, tz2));
	tmax = fminf(tmax, fmaxf(tz1, tz2));

	return tmax > fmaxf(tmin, 0.0f);
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

	if (hitBoundingBox(scene->boundingBox[0], ray)) {
		for (int i = 0; i < scene->numTriangles; i++) {
			if (hitTriangle(scene->triangles[i], ray, tMin, closestSoFar, &temp)) {
				hitAnything = true;
				closestSoFar = temp.t;
				*hitRecord = temp;
			}
		}
	}

	return hitAnything;
}