#pragma once

#include "ray.cuh"

struct HitRecord {
	float3 p;
	float3 normal;
	float t;
	bool frontFace;
	Material material;

	__device__ HitRecord() :
		p(make_float3(0.0f)),
		normal(make_float3(0.0f)),
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
	float3 h = cross(r.direction, t.edge2);
	float a = dot(t.edge1, h);

	if (fabsf(a) < 1e-8f)
		return false;

	float f = 1.0f / a;
	float3 s = r.origin - t.p0;
	float u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f)
		return false;

	float3 q = cross(s, t.edge1);
	float v = f * dot(r.direction, q);

	if (v < 0.0f || u + v > 1.0f)
		return false;

	float t0 = f * dot(t.edge2, q);

	if (t0 > tMin && t0 < tMax)	{
		hit->t = t0;
		hit->p = rayAt(r, t0);
		hit->material = t.material;
		setFaceNormal(hit, r, unit(t.smoothNormals ? (1.0f - u - v) * t.n0 + u * t.n1 + v * t.n2 : t.faceNormal));

		return true;
	}
	
	return false;
}

__device__ bool hitBoundingBox(const float3& min, const float3& max, const Ray& r) {
	float3 t1 = (min - r.origin) * r.invDirection;
	float3 t2 = (max - r.origin) * r.invDirection;

	float3 tmin3 = fminf3(t1, t2);
	float3 tmax3 = fmaxf3(t1, t2);

	float tmin = fmaxf(fmaxf(tmin3.x, tmin3.y), tmin3.z);
	float tmax = fminf(fminf(tmax3.x, tmax3.y), tmax3.z);

	return tmax > fmaxf(tmin, 0.0f);
}


__device__ bool hitBVH(BVHNode* nodes, Triangle* tris, int* triIndices, Ray& ray, float tMin, float tMax, HitRecord* outHit) {
	int stack[64];
	int sp = 0;
	stack[sp++] = 0;

	bool hitAnything = false;
	float closest = tMax;
	HitRecord tempHit;

	while (sp > 0) {
		BVHNode& node = nodes[stack[--sp]];

		if (!hitBoundingBox(node.min, node.max, ray))
			continue;

		if (node.count > 0) {
			for (int i = 0; i < node.count; ++i) {
				int triIdx = triIndices[node.leftFirst + i];
				if (hitTriangle(tris[triIdx], ray, tMin, closest, &tempHit)) {
					hitAnything = true;
					closest = tempHit.t;
					*outHit = tempHit;
				}
			}
		} else {
			if (node.leftFirst >= 0)
				stack[sp++] = node.leftFirst;
			if (node.rightChild >= 0)
				stack[sp++] = node.rightChild;
		}
	}

	return hitAnything;
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

	if (scene->numTriangles > 0) {
		if (hitBVH(scene->bvhNodes, scene->triangles, scene->triIndices, ray, tMin, closestSoFar, &temp)) {
			hitAnything = true;
			closestSoFar = temp.t;
			*hitRecord = temp;
		}
	}

	return hitAnything;
}