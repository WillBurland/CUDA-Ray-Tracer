#pragma once

#include <float.h>
#include <vector>

struct BVHNode {
	float3 min{FLT_MAX}, max{-FLT_MAX};
	int leftFirst{-1}, rightChild{-1};
	int count{0};
};

__host__ inline int getBVHDepth(const BVHNode* nodes, const int i) {
	const BVHNode& n = nodes[i];
	return n.count > 0 ? 1 : 1 + std::max(getBVHDepth(nodes, n.leftFirst), getBVHDepth(nodes, n.leftFirst + 1));
}

__host__ inline float getAxis(const float3& v, const int axis) {
	return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

__host__ inline int partition(std::vector<int>& triIndices, const int start, const int count, const int axis, const float splitPos, const std::vector<Triangle>& tris) {
	int i = start;
	int j = start + count - 1;
	while (i <= j) {
		const Triangle& t = tris[triIndices[i]];
		const float3 centroid = (t.p0 + (t.p0 + t.edge1) * 0.5f + (t.p0 + t.edge2) * 0.5f) / 3.0f;
		if (getAxis(centroid, axis) < splitPos)
			i++;
		else
			std::swap(triIndices[i], triIndices[j--]);
	}
	return i;
}

__host__ inline int buildBVH(BVHNode* nodes, int& nodeCount, std::vector<int>& triIndices, const int start, const int count, const std::vector<Triangle>& tris) {
	const int nodeIndex = nodeCount++;
	BVHNode& node = nodes[nodeIndex];

	float3 minB = make_float3(FLT_MAX);
	float3 maxB = make_float3(-FLT_MAX);
	for (int i = start; i < start + count; i++) {
		float3 triMin, triMax;
		tris[triIndices[i]].bounds(triMin, triMax);
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

	const int axis = maxComponent(node.max - node.min);
	const float splitPos = (getAxis(node.min, axis) + getAxis(node.max, axis)) * 0.5f;
	const int mid = partition(triIndices, start, count, axis, splitPos, tris);

	if (mid == start || mid == start + count) {
		node.leftFirst = start;
		node.count = count;
		return nodeIndex;
	}

	const int left = buildBVH(nodes, nodeCount, triIndices, start, mid - start, tris);
	const int right = buildBVH(nodes, nodeCount, triIndices, mid, count - (mid - start), tris);

	node.leftFirst = left;
	node.rightChild = right;
	node.count = 0;

	return nodeIndex;
}