#pragma once

__host__ __device__ inline float3 make_float3(const float a) {
	return {a, a, a};
}

__host__ __device__ inline float3 operator+(const float3 a, const float3 b) {
	return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ inline float3& operator+=(float3& a, const float3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

__host__ __device__ inline float3 operator-(const float3 a, const float3 b) {
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ inline float3 operator-(const float3 a) {
	return {-a.x, -a.y, -a.z};
}

__host__ __device__ inline float3 operator*(const float3 a, const float3 b) {
	return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ inline float3& operator*=(float3& a, const float3 b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

__host__ __device__ inline float3 operator*(const float3 a, const float b) {
	return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ inline float3 operator*(const float a, const float3 b) {
	return {b.x * a, b.y * a, b.z * a};
}

__host__ __device__ inline float3 operator/(const float3 a, const float b) {
	return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__ inline float3 operator/(const float a, const float3 b) {
	return {a / b.x, a / b.y, a / b.z};
}

__host__ __device__ inline float3& operator/=(float3& a, const float b) {
	a.x /= b;
	a.y /= b;
	a.z /= b;
	return a;
}

__host__ __device__ inline float dot(const float3 a, const float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(const float3 a, const float3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

__host__ __device__ inline float length2(const float3 a) {
	return a.x * a.x + a.y * a.y + a.z * a.z;
}

__host__ __device__ inline float length(const float3 a) {
	return sqrtf(length2(a));
}

__host__ __device__ inline float3 unit(const float3 a) {
	const float len = length(a);
	return len > 0.0f ? (a / len) : make_float3(0.0f);
}

__host__ __device__ inline float3 inv(const float3 a) {
	return {
		a.x != 0.0f ? 1.0f / a.x : 0.0f,
		a.y != 0.0f ? 1.0f / a.y : 0.0f,
		a.z != 0.0f ? 1.0f / a.z : 0.0f
	};
}

__host__ __device__ inline float3 reflect(const float3 v, const float3 n) {
	return v - n * 2.0f * dot(v, n);
}

__host__ __device__ inline float3 refract(const float3 uv, const float3 n, const float etaiOverEtat) {
	const float cosTheta = fminf(dot(uv * -1.0f, n), 1.0f);
	const float3 rOutPerp = etaiOverEtat * (uv + cosTheta * n);
	const float3 rOutParallel = (n * -1.0f) * sqrtf(fabsf(1.0f - length2(rOutPerp)));
	return rOutPerp + rOutParallel;
}

__host__ __device__ inline float reflectance(const float cosine, const float refIdx) {
	float r0 = (1 - refIdx) / (1 + refIdx);
	r0 *= r0;
	return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__host__ __device__ inline ulong nextSeed(const ulong seed) {
	return (seed * 0x5DEECE66D + 0xB) & ((1ULL << 48) - 1);
}

__host__ __device__ inline float randFloat(ulong* seed) {
	*seed = nextSeed(*seed);
	return static_cast<float>(*seed & 0xFFFFFF) / static_cast<float>(0x1000000);
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

__host__ __device__ inline bool nearZero(const float3 a) {
	const float s = 1e-8;
	return (fabsf(a.x) < s) && (fabsf(a.y) < s) && (fabsf(a.z) < s);
}

__host__ __device__ inline float2 uv(float3 n) {
	n = unit(n);
	return make_float2(
		0.5f + atan2(n.z, n.x) / (2.0f * static_cast<float>(M_PI)),
		0.5f - asin(n.y) / static_cast<float>(M_PI));
}

__host__ __device__ inline float3 clamp3(float3 a, const float min, const float max) {
	return make_float3(
		a.x = fminf(fmaxf(a.x, min), max),
		a.y = fminf(fmaxf(a.y, min), max),
		a.z = fminf(fmaxf(a.z, min), max)
	);
}

__host__ __device__ inline float maxComponent(const float3 a) {
	return fmaxf(fmaxf(a.x, a.y), a.z);
}

__host__ __device__ inline float3 fminf3(const float3& a, const float3& b) {
	return make_float3(
		fminf(a.x, b.x),
		fminf(a.y, b.y),
		fminf(a.z, b.z)
	);
}

__host__ __device__ inline float3 fmaxf3(const float3& a, const float3& b) {
	return make_float3(
		fmaxf(a.x, b.x),
		fmaxf(a.y, b.y),
		fmaxf(a.z, b.z)
	);
}

__host__ __device__ inline float3 powf3(const float3 a, const float b) {
	return make_float3(
		powf(a.x, b),
		powf(a.y, b),
		powf(a.z, b)
	);
}