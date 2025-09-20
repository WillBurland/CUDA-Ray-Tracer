#pragma once

struct Triangle {
	float3 p0, p1, p2;
	float3 n0, n1, n2;
	Material material;
	int boundingBoxId;

	Triangle() :
		p0(make_float3(0.0f, 0.0f, 0.0f)),
		p1(make_float3(0.0f, 0.0f, 0.0f)),
		p2(make_float3(0.0f, 0.0f, 0.0f)),
		n0(make_float3(0.0f, 0.0f, 0.0f)),
		n1(make_float3(0.0f, 0.0f, 0.0f)),
		n2(make_float3(0.0f, 0.0f, 0.0f)),
		material(Material()),
		boundingBoxId(0) {}

	Triangle(float3 p0, float3 p1, float3 p2, float3 n0, float3 n1, float3 n2, Material material, int boundingBoxId) :
		p0(p0),
		p1(p1),
		p2(p2),
		n0(n0),
		n1(n1),
		n2(n2),
		material(material),
		boundingBoxId(boundingBoxId) {}
};
