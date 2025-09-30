#pragma once

#include "bounding_box.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "triangle.cuh"
#include "bvh_node.cuh"

struct Scene {
	Material* materials;
	int numMaterials;
	Sphere* spheres;
	int numSpheres;
	Triangle* triangles;
	int numTriangles;
	BoundingBox* boundingBox;
	BVHNode* bvhNodes;
	int numBVHNodes;
	int* triIndices;
	cudaTextureObject_t hdrTex;
	int hdrImageWidth, hdrImageHeight;
	Camera camera;

	__host__ Scene(
		Material* materials, const int numMaterials,
		Sphere* spheres, const int numSpheres,
		Triangle* triangles, const int numTriangles,
		BoundingBox* boundingBox,
		BVHNode* bvhNodes, const int numBVHNodes,
		int* triIndices,
		const cudaTextureObject_t hdrTex,
		const int hdrImageWidth, const int hdrImageHeight,
		const Camera camera
	) :
		materials(materials),
		numMaterials(numMaterials),
		spheres(spheres),
		numSpheres(numSpheres),
		triangles(triangles),
		numTriangles(numTriangles),
		boundingBox(boundingBox),
		bvhNodes(bvhNodes),
		numBVHNodes(numBVHNodes),
		triIndices(triIndices),
		hdrTex(hdrTex),
		hdrImageWidth(hdrImageWidth),
		hdrImageHeight(hdrImageHeight),
		camera(camera) {}
};
