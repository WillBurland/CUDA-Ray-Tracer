#pragma once

#include "ray.cuh"

struct HitRecord {
	float3 hitPoint{make_float3(0.0f)};
	float3 normal{make_float3(0.0f)};
	float hitDistance{0.0f};
	bool frontFace{false};
	Material material{};

	__device__ void setFaceNormal(const Ray& ray, const float3& outwardNormal) {
		this->frontFace = dot(ray.direction, outwardNormal) < 0;
		this->normal = frontFace ? outwardNormal : -outwardNormal;
	}

	__device__ bool hitSphere(const Sphere s, const Ray r, const float tMin, const float tMax, const Material* materials) {
		const float3 oc = r.origin - s.centre;
		const float halfB = dot(oc, r.direction);
		const float discriminant = halfB * halfB - (length2(oc) - s.radius * s.radius);

		if (discriminant < 0)
			return false;

		const float sqrtD = sqrtf(discriminant);
		float root = -halfB - sqrtD;
		if (root < tMin || tMax < root) {
			root = -halfB + sqrtD;
			if (root < tMin || tMax < root)
				return false;
		}

		this->hitDistance = root;
		this->hitPoint = r.at(root);
		this->material = materials[s.materialId];
		setFaceNormal(r, (hitPoint - s.centre) / s.radius);

		return true;
	}

	__device__ bool hitTriangle(const Triangle tri, const Ray r, const float tMin, const float tMax, const Material* materials) {
		const float3 h = cross(r.direction, tri.edge2);
		const float a = dot(tri.edge1, h);

		if (fabsf(a) < 1e-8f)
			return false;

		const float f = 1.0f / a;
		const float3 s = r.origin - tri.p0;
		const float u = f * dot(s, h);

		if (u < 0.0f || u > 1.0f)
			return false;

		const float3 q = cross(s, tri.edge1);
		const float v = f * dot(r.direction, q);

		if (v < 0.0f || u + v > 1.0f)
			return false;

		const float t0 = f * dot(tri.edge2, q);

		if (t0 > tMin && t0 < tMax)	{
			this->hitDistance = t0;
			this->hitPoint = r.at(t0);
			this->material = materials[tri.materialId];
			setFaceNormal(r, unit(tri.smoothNormals ? (1.0f - u - v) * tri.n0 + u * tri.n1 + v * tri.n2 : tri.faceNormal));

			return true;
		}
		
		return false;
	}

	__device__ bool hitBVH(const BVHNode* nodes, const Triangle* tris, const int* triIndices, const Ray& ray, const float tMin, const float tMax, const Material* materials) {
		int stack[64];
		int sp = 0;
		stack[sp++] = 0;

		bool hitAnything = false;
		float closest = tMax;
		HitRecord tempHit;

		while (sp > 0) {
			const BVHNode& node = nodes[stack[--sp]];

			const float3 t1 = (node.min - ray.origin) * ray.invDirection;
			const float3 t2 = (node.max - ray.origin) * ray.invDirection;

			const float3 tmin3 = fminf3(t1, t2);
			const float3 tmax3 = fmaxf3(t1, t2);

			const float tmin = fmaxf(fmaxf(tmin3.x, tmin3.y), tmin3.z);
			const float tmax = fminf(fminf(tmax3.x, tmax3.y), tmax3.z);

			if (tmax <= fmaxf(tmin, 0.0f))
				continue;

			if (node.count > 0) {
				for (int i = 0; i < node.count; ++i) {
					if (tempHit.hitTriangle(tris[triIndices[node.leftFirst + i]], ray, tMin, closest, materials)) {
						hitAnything = true;
						closest = tempHit.hitDistance;
						*this = tempHit;
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

	__device__ bool hitAnything(const Ray ray, const float tMin, const float tMax, const Scene* scene) {
		HitRecord temp;
		bool hitAnything = false;
		float closestSoFar = INFINITY;

		for (int i = 0; i < scene->numSpheres; i++) {
			if (temp.hitSphere(scene->spheres[i], ray, tMin, closestSoFar, scene->materials)) {
				hitAnything = true;
				closestSoFar = temp.hitDistance;
				*this = temp;
			}
		}

		if (scene->numTriangles > 0) {
			if (temp.hitBVH(scene->bvhNodes, scene->triangles, scene->triIndices, ray, tMin, closestSoFar, scene->materials)) {
				hitAnything = true;
				closestSoFar = temp.hitDistance;
				*this = temp;
			}
		}

		return hitAnything;
	}
};