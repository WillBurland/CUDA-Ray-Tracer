#pragma once

#include <float.h>
#include <vector>

struct BVHNode {
	float3 min, max;
	int leftFirst;
	int rightChild;
	int count;

	BVHNode() :
		min(make_float3(0.0f)),
		max(make_float3(0.0f)),
		leftFirst(-1),
		rightChild(-1),
		count(0) {}

	BVHNode(float3 min, float3 max, int leftFirst, int rightChild,  int count) :
		min(min),
		max(max),
		leftFirst(leftFirst),
		rightChild(rightChild),
		count(count) {}
};

inline int getBVHDepth(BVHNode* nodes, int i) {
	const BVHNode& n = nodes[i];
	return n.count > 0 ? 1 : 1 + std::max(getBVHDepth(nodes, n.leftFirst), getBVHDepth(nodes, n.leftFirst + 1));
}

inline float getAxis(float3 &v, int axis) {
	return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

inline void triangleBounds(Triangle &t, float3 &minOut, float3 &maxOut) {
	float3 p1 = t.p0 + t.edge1;
	float3 p2 = t.p0 + t.edge2;
	minOut = fminf3(t.p0, fminf3(p1, p2));
	maxOut = fmaxf3(t.p0, fmaxf3(p1, p2));
}

inline int partition(std::vector<int> &triIndices, int start, int count, int axis, float splitPos, std::vector<Triangle> &tris) {
	int i = start, j = start + count - 1;
	while (i <= j) {
		Triangle& t = tris[triIndices[i]];
		float3 centroid = (t.p0 + (t.p0 + t.edge1) * 0.5f + (t.p0 + t.edge2) * 0.5f) / 3.0f;
		if (getAxis(centroid, axis) < splitPos)
			i++;
		else
			std::swap(triIndices[i], triIndices[j--]);
	}
	return i;
}

inline int buildBVH(BVHNode* nodes, int &nodeCount, std::vector<int> &triIndices, int start, int count, std::vector<Triangle> &tris) {
	int nodeIndex = nodeCount++;
	BVHNode& node = nodes[nodeIndex];

	float3 minB = make_float3(FLT_MAX);
	float3 maxB = make_float3(-FLT_MAX);
	for (int i = start; i < start + count; i++) {
		float3 triMin, triMax;
		triangleBounds(tris[triIndices[i]], triMin, triMax);
		minB = fminf3(minB, triMin);
		maxB = fmaxf3(maxB, triMax);
	}
	node.min = minB;
	node.max = maxB;

	if (count <= 8) {
		node.leftFirst = start;
		node.count = count;
		return nodeIndex;
	}

	int axis = maxComponent(node.max - node.min);
	float splitPos = (getAxis(node.min, axis) + getAxis(node.max, axis)) * 0.5f;
	int mid = partition(triIndices, start, count, axis, splitPos, tris);

	if (mid == start || mid == start + count) {
		node.leftFirst = start;
		node.count = count;
		return nodeIndex;
	}

	int left  = buildBVH(nodes, nodeCount, triIndices, start, mid - start, tris);
	int right = buildBVH(nodes, nodeCount, triIndices, mid, count - (mid - start), tris);

	node.leftFirst = left;
	node.rightChild = right;
	node.count = 0;

	return nodeIndex;
}