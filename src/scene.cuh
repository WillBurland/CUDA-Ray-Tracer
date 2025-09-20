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
	Camera camera;

	Scene(Sphere* spheres, int numSpheres, Triangle* triangles, int numTriangles, BoundingBox* boundingBox, Camera camera) :
		spheres(spheres),
		numSpheres(numSpheres),
		triangles(triangles),
		numTriangles(numTriangles),
		boundingBox(boundingBox),
		camera(camera) {}
};