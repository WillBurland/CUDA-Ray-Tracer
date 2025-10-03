#include <fstream>
#include <nlohmann/json.hpp>

#include "camera.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "triangle.cuh"
#include "bounding_box.cuh"
#include "bvh_node.cuh"
#include "scene.cuh"

struct MeshData {
	std::string obj_path;
	float3 translation;
	float3 scale;
	float3 rotation;
	int materialId;
};

struct HostSceneData {
	std::vector<Material> materials;
	std::vector<Sphere> spheres;
	std::vector<Triangle> triangles;
	BoundingBox* boundingBox = nullptr;
	BVHNode* bvhNodes = nullptr;
	int numBVHNodes = 0;
	int* triIndices = nullptr;
	cudaTextureObject_t hdrTex = 0;
	int hdrImageWidth = 0;
	int hdrImageHeight = 0;
	Camera camera;
	MeshData mesh;

	int imageWidth = 800;
	int imageHeight = 600;
	int samplesPerPixel = 5;
	int maxBounces = 5;
	std::string hdriPath;

	__host__ HostSceneData() {};
};

__host__ inline void checkCuda(const cudaError_t err, const std::string msg) {
	if (err != cudaSuccess)
		throw std::runtime_error(msg + ": " + cudaGetErrorString(err));
}

template<typename T>
struct CudaBuffer {
	T* ptr = nullptr;
	size_t size = 0;

	__host__ void allocate(size_t n) {
		size = n;
		checkCuda(cudaMalloc(&ptr, n * sizeof(T)), "malloc");
	}

	__host__ void copyFromHost(const T* src) {
		checkCuda(cudaMemcpy(ptr, src, size * sizeof(T), cudaMemcpyHostToDevice), "memcpy");
	}

	__host__ ~CudaBuffer() {
		if (ptr)
			cudaFree(ptr);
	}
};

template<typename T>
__host__ inline T require(const nlohmann::json& j, const std::string& key, const std::string& path) {
	if (!j.contains(key))
		throw std::runtime_error("Missing '" + key + "' on " + path);
	return j.at(key).get<T>();
}

__host__ inline float3 loadFloat3(const nlohmann::json& j, const std::string& v1, const std::string& v2, const std::string& v3, const std::string& path) {
	return {
		require<float>(j, v1, path),
		require<float>(j, v2, path),
		require<float>(j, v3, path),
	};
}

__host__ inline Camera loadCamera(const nlohmann::json& j, const float aspectRatio) {
	return {
		loadFloat3(j.at("position"), "x", "y", "z", "camera.position"),
		loadFloat3(j.at("look_at"),  "x", "y", "z", "camera.look_at"),
		loadFloat3(j.at("up"),       "x", "y", "z", "camera.up"),
		require<float>(j, "fov", "camera"),
		require<float>(j, "focus_distance", "camera"),
		require<float>(j, "f_stop", "camera"),
		aspectRatio
	};
}

__host__ inline MeshData loadMesh(const nlohmann::json& j) {
	return {
		require<std::string>(j, "obj_path", "mesh"),
		loadFloat3(j.at("translation"), "x", "y", "z", "mesh.translation"),
		loadFloat3(j.at("scale"),       "x", "y", "z", "mesh.scale"),
		loadFloat3(j.at("rotation"),    "x", "y", "z", "mesh.rotation"),
		require<int>(j, "materialId", "mesh")
	};
}

__host__ inline std::vector<Sphere> loadSpheres(const nlohmann::json& arr) {
	std::vector<Sphere> result;
	for (const auto& s : arr) {
		result.emplace_back(
			loadFloat3(s.at("position"), "x", "y", "z", "spheres[?].position"),
			require<float>(s, "radius", "spheres[?]"),
			require<int>(s, "materialId", "spheres[?]")
		);
	}
	return result;
}

__host__ inline std::vector<Material> loadMaterials(const nlohmann::json& arr) {
	std::vector<Material> result;
	for (const auto& m : arr) {
		const int id = require<int>(m, "id", "materials");
		const std::string type = require<std::string>(m, "type", "materials[?]");

		if (type == "lambertian") {
			result.emplace_back(Material::Lambertian(
				id,
				loadFloat3(m.at("albedo"), "r", "g", "b", "materials[?].albedo")
			));
		} else if (type == "metal") {
			result.emplace_back(Material::Metal(
				id,
				loadFloat3(m.at("albedo"), "r", "g", "b", "materials[?].albedo"),
				require<float>(m, "fuzz", "materials[?]")
			));
		} else if (type == "transparent") {
			result.emplace_back(Material::Transparent(
				id,
				require<float>(m, "ior","materials[?].ior")
			));
		} else if (type == "emissive") {
			result.emplace_back(Material::Emissive(
				id,
				loadFloat3(m.at("albedo"), "r", "g", "b", "materials[?].albedo"),
				require<float>(m, "power", "materials[?]")
			));
		} else {
			throw std::runtime_error("Unknown material type: " + type);
		}
	}
	return result;
}

__host__ inline void loadHDRI(const nlohmann::json& settings, HostSceneData& scene) {
	const std::string path = require<std::string>(settings, "hdri_path", "settings");
	int w, h, c;
	const float* data = stbi_loadf(path.c_str(), &w, &h, &c, 3);
	if (!data)
		throw std::runtime_error("Unable to open HDRI: " + path);

	const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &desc, w, h);

	std::vector<float4> pixels(w * h);
	for (int i = 0; i < w * h; i++) {
		pixels[i] = {
			data[i * 3 + 0],
			data[i * 3 + 1],
			data[i * 3 + 2],
			1.0f
		};
	}

	cudaMemcpy2DToArray(cuArray, 0, 0, pixels.data(), w * sizeof(float4), w * sizeof(float4), h, cudaMemcpyHostToDevice);

	cudaResourceDesc res{};
	res.resType = cudaResourceTypeArray;
	res.res.array.array = cuArray;
	cudaTextureDesc tex{};
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModeLinear;
	tex.readMode = cudaReadModeElementType;
	tex.normalizedCoords = 1;

	cudaCreateTextureObject(&scene.hdrTex, &res, &tex, nullptr);
	scene.hdrImageWidth = w;
	scene.hdrImageHeight = h;
}

__host__ inline void loadSettings(const nlohmann::json& settings, HostSceneData& scene) {
	scene.imageWidth = require<int>(settings, "width", "settings");
	scene.imageHeight = require<int>(settings, "height", "settings");
	scene.samplesPerPixel = require<int>(settings, "samples_per_pixel", "settings");
	scene.maxBounces = require<int>(settings, "max_bounces", "settings");
	scene.hdriPath = require<std::string>(settings, "hdri_path", "settings");
	loadHDRI(settings, scene);
}

__host__ inline HostSceneData loadSceneDescription(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open())
		throw std::runtime_error("Failed to open scene file: " + path);

	nlohmann::json jsonData;
	try {
		file >> jsonData;
	} catch (const nlohmann::json::parse_error& e) {
		throw std::runtime_error(std::string("JSON parse error: ") + e.what());
	}

	HostSceneData scene;
	try {
		loadSettings(jsonData.at("settings"), scene);
		scene.camera = loadCamera(jsonData.at("camera"), static_cast<float>(scene.imageWidth) / static_cast<float>(scene.imageHeight));
		scene.materials = loadMaterials(jsonData.at("materials"));
		scene.spheres = loadSpheres(jsonData.at("spheres"));
		scene.mesh = loadMesh(jsonData.at("mesh"));
	} catch (const nlohmann::json::out_of_range& e) {
		throw std::runtime_error(std::string("Missing JSON field: ") + e.what());
	}

	return scene;
}