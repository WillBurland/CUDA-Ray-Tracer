#pragma once

struct Triangle {
	float3 p0{make_float3(0.0f)}, edge1{make_float3(0.0f)}, edge2{make_float3(0.0f)};
	float3 n0{make_float3(0.0f)}, n1{make_float3(0.0f)}, n2{make_float3(0.0f)};
	float3 faceNormal{make_float3(0.0f)};
	int materialId{-1};
	int boundingBoxId{-1};
	bool smoothNormals{false};

	__host__ Triangle() :
		p0(make_float3(0.0f)),
		edge1(make_float3(0.0f)),
		edge2(make_float3(0.0f)),
		n0(make_float3(0.0f)),
		n1(make_float3(0.0f)),
		n2(make_float3(0.0f)),
		faceNormal(make_float3(0.0f)),
		materialId(-1),
		boundingBoxId(-1),
		smoothNormals(false) {}

	__host__ Triangle(const float3 p0, const float3 p1, const float3 p2, const float3 n0, const float3 n1, const float3 n2, const int materialId, const int boundingBoxId, const bool smoothNormals) :
		p0(p0),
		edge1(p1 - p0),
		edge2(p2 - p0),
		n0(n0),
		n1(n1),
		n2(n2),
		faceNormal(cross(edge1, edge2)),
		materialId(materialId),
		boundingBoxId(boundingBoxId),
		smoothNormals(smoothNormals) {}

	__host__ inline void bounds(float3& minOut, float3& maxOut) const {
		const float3 p1 = p0 + edge1;
		const float3 p2 = p0 + edge2;
		minOut = fminf3(p0, fminf3(p1, p2));
		maxOut = fmaxf3(p0, fmaxf3(p1, p2));
	}
};
