#pragma once

#include "material.cuh"

struct Triangle {
	float3 p0, edge1, edge2;
	float3 n0, n1, n2;
	float3 faceNormal;
	Material material;
	int boundingBoxId;
	bool smoothNormals;

	Triangle() :
		p0(make_float3(0.0f)),
		edge1(make_float3(0.0f)),
		edge2(make_float3(0.0f)),
		n0(make_float3(0.0f)),
		n1(make_float3(0.0f)),
		n2(make_float3(0.0f)),
		faceNormal(make_float3(0.0f)),
		material(Material()),
		boundingBoxId(0),
		smoothNormals(false) {}

	Triangle(float3 p0, float3 p1, float3 p2, float3 n0, float3 n1, float3 n2, Material material, int boundingBoxId, bool smoothNormals) :
		p0(p0),
		edge1(p1 - p0),
		edge2(p2 - p0),
		n0(n0),
		n1(n1),
		n2(n2),
		faceNormal(cross(edge1, edge2)),
		material(material),
		boundingBoxId(boundingBoxId),
		smoothNormals(smoothNormals) {}
};
