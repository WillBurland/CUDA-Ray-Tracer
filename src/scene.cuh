#pragma once

#include "bounding_box.cuh"
#include "camera.cuh"
#include "sphere.cuh"
#include "triangle.cuh"

struct Scene {
	Sphere* spheres;
	int numSpheres;
	Triangle* triangles;
	int numTriangles;
	BoundingBox* boundingBox;
	cudaTextureObject_t hdrTex;
	int hdrImageWidth, hdrImageHeight;
	Camera camera;

	Scene(Sphere* spheres, int numSpheres, Triangle* triangles, int numTriangles, BoundingBox* boundingBox, cudaTextureObject_t hdrTex, int hdrImageWidth, int hdrImageHeight, Camera camera) :
		spheres(spheres),
		numSpheres(numSpheres),
		triangles(triangles),
		numTriangles(numTriangles),
		boundingBox(boundingBox),
		hdrTex(hdrTex),
		hdrImageWidth(hdrImageWidth),
		hdrImageHeight(hdrImageHeight),
		camera(camera) {}
};