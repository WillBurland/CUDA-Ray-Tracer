#pragma once

#include "bounding_box.cuh"
#include "camera.cuh"
#include "sphere.cuh"
#include "triangle.cuh"
#include "bvh_node.cuh"

struct Scene {
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

	Scene(
		Sphere* spheres, int numSpheres,
		Triangle* triangles, int numTriangles,
		BoundingBox* boundingBox,
		BVHNode* bvhNodes, int numBVHNodes,
		int* triIndices,
		cudaTextureObject_t hdrTex,
		int hdrImageWidth, int hdrImageHeight,
		Camera camera
	) :
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
