#include "renderer.cuh"

#include "hit_record.cuh"

#include <cstdio>
#include <thread>

__device__ __managed__ unsigned int d_pixelsDone = 0;

__device__ bool lambertianScatter(const Ray ray, const HitRecord* hitRecord, float3* attenuation, Ray* scattered, ulong* seed) {
	float3 scatterDir = hitRecord->normal + randUnitVec(seed);
	if (nearZero(scatterDir))
		scatterDir = hitRecord->normal;

	scattered->origin = hitRecord->hitPoint;
	scattered->direction = scatterDir;
	scattered->invDirection = inv(scatterDir);
	*attenuation = hitRecord->material.albedo;
	return true;
}

__device__ bool metalScatter(const Ray ray, const HitRecord* hitRecord, float3* attenuation, Ray* scattered, ulong* seed) {
	const float3 reflected = reflect(unit(ray.direction), hitRecord->normal);
	scattered->origin = hitRecord->hitPoint;
	scattered->direction = hitRecord->material.fuzz > 0.0f ? reflected + randVecInUnitSphere(seed) * hitRecord->material.fuzz : reflected;
	scattered->invDirection = inv(scattered->direction);
	*attenuation = hitRecord->material.albedo;
	return dot(scattered->direction, hitRecord->normal) > 0;
}

__device__ bool transparentScatter(const Ray ray, const HitRecord* hitRecord, float3* attenuation, Ray* scattered, ulong* seed) {
	const float refractionRatio = hitRecord->frontFace ? (1.0f / hitRecord->material.ior) : hitRecord->material.ior;
	const float3 unitDirection = unit(ray.direction);
	const float cosTheta = fminf(dot(unitDirection * -1, hitRecord->normal), 1.0f);

	float3 direction;
	if (refractionRatio * sqrtf(1.0f - cosTheta * cosTheta) > 1.0f || reflectance(cosTheta, refractionRatio) > randFloat(seed)) {
		direction = reflect(unitDirection, hitRecord->normal);
	} else {
		direction = refract(unitDirection, hitRecord->normal, refractionRatio);
	}

	scattered->origin = hitRecord->hitPoint;
	scattered->direction = direction;
	scattered->invDirection = inv(scattered->direction);
	*attenuation = make_float3(1.0f);
	return true;
}

__device__ float3 rayColour(Ray ray, const Scene* scene, ulong* seed) {
	float3 rayColour = make_float3(1.0f);
	int currentBounces = 0;
	HitRecord hitRecord;

	while (currentBounces < scene->maxBounces) {
		if (hitRecord.hitAnything(ray, 0.001f, INFINITY, scene)) {
			Ray scattered(ray.origin, ray.direction, ray.invDirection);
			float3 attenuation;
			switch (hitRecord.material.type) {
				case LAMBERTIAN: {
					if (lambertianScatter(ray, &hitRecord, &attenuation, &scattered, seed)) {
						ray = scattered;
						rayColour *= attenuation;
						currentBounces++;
						continue;
					}
					return make_float3(0.0f);
				}
				case METAL: {
					if (metalScatter(ray, &hitRecord, &attenuation, &scattered, seed)) {
						ray = scattered;
						rayColour *= attenuation;
						currentBounces++;
						continue;
					}
					return make_float3(0.0f);
				}
				case TRANSPARENT: {
					if (transparentScatter(ray, &hitRecord, &attenuation, &scattered, seed)) {
						ray = scattered;
						rayColour *= attenuation;
						currentBounces++;
						continue;
					}
					return hitRecord.material.albedo;
				}
				case EMISSIVE: {
					return hitRecord.material.albedo;
				default: {
					return make_float3(1.0f, 0.0f, 1.0f);
				}
				}
			}
			continue;
		}
		break;
	}

	const float2 uvCoords = uv(ray.direction);
	const float4 texColour = tex2D<float4>(scene->hdrTex, uvCoords.x, uvCoords.y);
	
	return rayColour * make_float3(texColour.x, texColour.y, texColour.z);
}

__global__ void shadePixel(unsigned char* image, const Scene* scene) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= scene->imageWidth || y >= scene->imageHeight)
		return;

	const int idx = (scene->imageHeight - 1 - y) * scene->imageWidth + x;
	ulong seed = nextSeed((ulong)(idx * idx));

	float3 pixelColour = make_float3(0.0f);

	for (int i = 0; i < scene->samplesPerPixel; i++) {
		float3 colourToAdd = rayColour(
			Ray(
				scene->camera,
				(x + randFloat(&seed)) / scene->imageWidth,
				(y + randFloat(&seed)) / scene->imageHeight,
				&seed
			),
			scene,
			&seed
		);

		if (isnan(colourToAdd.x)) colourToAdd.x = 0.0f;
		if (isnan(colourToAdd.y)) colourToAdd.y = 0.0f;
		if (isnan(colourToAdd.z)) colourToAdd.z = 0.0f;

		pixelColour += colourToAdd;
	}

	pixelColour = clamp3(powf3(pixelColour / scene->samplesPerPixel, 1.0f / 2.2f), 0.0f, 1.0f);

	image[idx * 3 + 0] = pixelColour.x * 255.0f;
	image[idx * 3 + 1] = pixelColour.y * 255.0f;
	image[idx * 3 + 2] = pixelColour.z * 255.0f;

	atomicAdd(&d_pixelsDone, 1);
}

__host__ void renderImage(unsigned char* image, const Scene* scene, const int width, const int height) {
	const dim3 block(16, 16);
	const dim3 grid(
		(width  + block.x - 1) / block.x,
		(height + block.y - 1) / block.y
	);

	d_pixelsDone = 0;

	shadePixel<<<grid, block>>>(image, scene);
	
	const unsigned int totalPixels = width * height;
	const auto startTime = std::chrono::steady_clock::now();
	int lastLineLength = 0;
	while (true) {
		const float progress = 100.0f * (float)(d_pixelsDone + 1) / totalPixels;
		const float etaSec = std::chrono::duration_cast<std::chrono::seconds>(
			std::chrono::steady_clock::now() - startTime).count() * (100.0f / progress - 1.0f);

		char buffer[128];
		const int n = snprintf(buffer, sizeof(buffer), "Rendering: %.2f%% | ETA: %.1fs", progress, etaSec);

		if (n < lastLineLength) {
			for (int i = n; i < lastLineLength; ++i)
				buffer[i] = ' ';
			buffer[lastLineLength] = '\0';
		}
		lastLineLength = n;

		printf("\r%s", buffer);
		fflush(stdout);

		if (d_pixelsDone >= totalPixels)
			break;

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}

	cudaDeviceSynchronize();
	printf("\n");
}