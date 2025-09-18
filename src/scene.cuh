#pragma once

#include "camera.cuh"
#include "sphere.cuh"

struct Scene {
	Sphere* spheres;
	int numSpheres;
	Camera camera;

	Scene(Sphere* spheres, int numSpheres, Camera camera) :
		spheres(spheres),
		numSpheres(numSpheres),
		camera(camera) {}
};