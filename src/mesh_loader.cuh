#pragma once

#include "bounding_box.cuh"
#include "triangle.cuh"

#include <float.h>
#include <fstream>
#include <sstream>
#include <vector>

__host__ float3 rotate(const float3& v, const float3& rotDeg) {
	float3 vr = v;

	const float degToRad = 3.14159265f / 180.0f;
	const float cx = cosf(rotDeg.x * degToRad);
	const float sx = sinf(rotDeg.x * degToRad);
	const float cy = cosf(rotDeg.y * degToRad);
	const float sy = sinf(rotDeg.y * degToRad);
	const float cz = cosf(rotDeg.z * degToRad);
	const float sz = sinf(rotDeg.z * degToRad);

	const float y1 = vr.y * cx - vr.z * sx;
	const float z1 = vr.y * sx + vr.z * cx;
	vr.y = y1;
	vr.z = z1;

	const float x2 = vr.x * cy + vr.z * sy;
	const float z2 = -vr.x * sy + vr.z * cy;
	vr.x = x2;
	vr.z = z2;

	const float x3 = vr.x * cz - vr.y * sz;
	const float y3 = vr.x * sz + vr.y * cz;
	vr.x = x3;
	vr.y = y3;

	return vr;
}

__host__ void loadTriangles(std::ifstream& meshFile, const float3 scaling, const float3 translation, const float3 rotation, const int materialId, Triangle*& h_triangles, int& numTriangles, BoundingBox*& h_boundingBox, const bool smoothNormals) {
	std::vector<float3> vertices, normals;
	std::vector<Triangle> triangles;
	std::string line;
	float3 min = make_float3(FLT_MAX);
	float3 max = make_float3(-FLT_MAX);

	while (std::getline(meshFile, line)) {
		std::istringstream iss(line);
		std::string type;
		iss >> type;

		if (type == "v") {
			float x, y, z;
			iss >> x >> y >> z;

			const float3 v = (rotate(make_float3(x, y, z), rotation) * scaling) + translation;
			vertices.push_back(v);

			min.x = std::min(min.x, v.x);
			min.y = std::min(min.y, v.y);
			min.z = std::min(min.z, v.z);

			max.x = std::max(max.x, v.x);
			max.y = std::max(max.y, v.y);
			max.z = std::max(max.z, v.z);
		} else if (type == "vn") {
			float x, y, z;
			iss >> x >> y >> z;
			normals.push_back(unit(rotate(make_float3(x, y, z), rotation)));
		} else if (type == "f") {
			std::vector<int> faceIndices;
			std::vector<int> faceNormals;
			std::string vertSpec;

			while (iss >> vertSpec) {
				int vi = 0, ni = 0;
				const int firstSlash = vertSpec.find('/');
				const int secondSlash = vertSpec.find('/', firstSlash + 1);

				if (firstSlash == -1) {
					vi = std::stoi(vertSpec);
					ni = -1;
				} else if (secondSlash == -1) {
					vi = std::stoi(vertSpec.substr(0, firstSlash));
					ni = -1;
				} else {
					vi = std::stoi(vertSpec.substr(0, firstSlash));
					ni = std::stoi(vertSpec.substr(secondSlash + 1));
				}

				vi = vi > 0 ? vi - 1 : vi + (int)vertices.size();
				if (ni != -1)
					ni = ni > 0 ? ni - 1 : ni + (int)normals.size();

				faceIndices.push_back(vi);
				faceNormals.push_back(ni);
			}

			for (int i = 1; i + 1 < faceIndices.size(); i++) {
				const float3 n0 = faceNormals[0] 	 != -1 ? normals[faceNormals[0]] :
					unit(cross(
						vertices[faceIndices[1]] - vertices[faceIndices[0]],
						vertices[faceIndices[2]] - vertices[faceIndices[0]]));
				const float3 n1 = faceNormals[i]   != -1 ? normals[faceNormals[i]]   : n0;
				const float3 n2 = faceNormals[i+1] != -1 ? normals[faceNormals[i+1]] : n0;

				triangles.push_back(Triangle(
					vertices[faceIndices[0]],
					vertices[faceIndices[i]],
					vertices[faceIndices[i + 1]],
					n0,
					n1,
					n2,
					materialId,
					0,
					smoothNormals
				));
			}
		}
	}

	numTriangles = (int)triangles.size();
	h_triangles = new Triangle[numTriangles];
	std::copy(triangles.begin(), triangles.end(), h_triangles);
	h_boundingBox = new BoundingBox(0, -1, min, max);
}