#pragma once

__host__ __device__ inline float3 operator+(float3 a, float3 b) {
	return make_float3(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z
	);
}

__host__ __device__ inline float3& operator+=(float3& a, float3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

__host__ __device__ inline float3 operator-(float3 a, float3 b) {
	return make_float3(
		a.x - b.x,
		a.y - b.y,
		a.z - b.z
	);
}

__host__ __device__ inline float3 operator*(float3 a, float3 b) {
	return make_float3(
		a.x * b.x,
		a.y * b.y,
		a.z * b.z
	);
}

__host__ __device__ inline float3& operator*=(float3& a, float3 b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

__host__ __device__ inline float3 operator*(float3 a, float b) {
	return make_float3(
		a.x * b,
		a.y * b,
		a.z * b
	);
}

__host__ __device__ inline float3 operator*(float a, float3 b) {
	return make_float3(
		b.x * a,
		b.y * a,
		b.z * a
	);
}

__host__ __device__ inline float3 operator/(float3 a, float b) {
	return make_float3(
		a.x / b,
		a.y / b,
		a.z / b
	);
}

__host__ __device__ inline float3 operator/(float a, float3 b) {
	return make_float3(
		a / b.x,
		a / b.y,
		a / b.z
	);
}

__host__ __device__ inline float dot(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(float3 a, float3 b) {
	return make_float3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

__host__ __device__ inline float length2(float3 a) {
	return a.x * a.x + a.y * a.y + a.z * a.z;
}

__host__ __device__ inline float length(float3 a) {
	return sqrtf(length2(a));
}

__host__ __device__ inline float3 unit(float3 a) {
	float len = length(a);
	return len > 0.0f ? (a / len) : make_float3(0.0f, 0.0f, 0.0f);
}

__host__ __device__ inline float3 inv(float3 a) {
	return make_float3(
		a.x != 0.0f ? 1.0f / a.x : 0.0f,
		a.y != 0.0f ? 1.0f / a.y : 0.0f,
		a.z != 0.0f ? 1.0f / a.z : 0.0f
	);
}

__host__ __device__ inline float3 reflect(float3 v, float3 n) {
	return v - n * 2.0f * dot(v, n);
}

__host__ __device__ inline float3 refract(float3 uv, float3 n, float etaiOverEtat) {
	float cosTheta = fminf(dot(uv * -1.0f, n), 1.0f);
	float3 rOutPerp = etaiOverEtat * (uv + cosTheta * n);
	float3 rOutParallel = (n * -1.0f) * sqrtf(fabsf(1.0f - length2(rOutPerp)));
	return rOutPerp + rOutParallel;
}

__host__ __device__ inline float reflectance(float cosine, float refIdx) {
	float r0 = (1 - refIdx) / (1 + refIdx);
	r0 *= r0;
	return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__host__ __device__ inline ulong nextSeed(ulong seed) {
	return (seed * 0x5DEECE66D + 0xB) & ((1ULL << 48) - 1);
}

__host__ __device__ inline float randFloat(ulong* seed) {
	*seed = nextSeed(*seed);
	return (float)(*seed & 0xFFFFFF) / (float)0x1000000;
}

__host__ __device__ inline float3 randVecInUnitSphere(ulong* seed) {
	float3 res;
	while (true) {
		res = make_float3(
			randFloat(seed) * 2.0f - 1.0f,
			randFloat(seed) * 2.0f - 1.0f,
			randFloat(seed) * 2.0f - 1.0f
		);
		if (length2(res) < 1.0f)
			return res;
	}	
}

__host__ __device__ inline float3 randVecInUnitDisk(ulong* seed) {
	float3 res;
	while (true) {
		res = make_float3(
			randFloat(seed) * 2.0f - 1.0f,
			randFloat(seed) * 2.0f - 1.0f,
			0.0f
		);
		if (length2(res) < 1.0f)
			return res;
	}	
}

__host__ __device__ inline float3 randUnitVec(ulong* seed) {
	return unit(randVecInUnitSphere(seed));
}

__host__ __device__ inline bool nearZero(float3 a) {
	float s = 1e-8;
	return (fabsf(a.x) < s) && (fabsf(a.y) < s) && (fabsf(a.z) < s);
}

__host__ __device__ inline float2 uv(float3 n) {
	n = unit(n);
	return make_float2(
		0.5f + atan2(n.z, n.x) / (2.0f * (float)M_PI),
		0.5f - asin(n.y) / (float)M_PI);
}