# CUDA Ray Tracer
	
## About the project

As the 4th iteration of my raytracing project, I wanted to get an even higher level of computational performance. OpenCL has the benefit of having cross vendor support and more flexibility, however that comes at a performance cost due to the more generalised nature of the compiled code. CUDA on the other hand is a specialised framework for Nvidia GPUs which can extract more compute power from the same code, at the cost of vendor lock-in.

## Sample image from a recent commit

None, while I get this project up to feature parity with the OpenCL project.

## Comparison to previous projects

Note: This isn't _really_ a fair comparison due to the varying feature set, complexity and my personal knowledge between the projects, but serves as a rough benchmark.

### Output variables

| Attribute         | Value      |
|-------------------|------------|
| Output resolution | 1280 x 720 |
| Samples per pixel | 250        |
| Max bounce depth  | 50         |

### Scene environment

| Sphere # | X, Y, Z           | Radius | R, G, B       | Material   | Fuzz |
|----------|-------------------|--------|---------------|------------|------|
| 1        | 0.0, -100.5, -1.0 | 100.0  | 0.0, 0.8, 0.7 | Lambertian | N/A  |
| 2        | 0.0, 0.5, -1.0    | 0.5    | 0.7, 0.3, 0.9 | Lambertian | N/A  |
| 3        | -0.9, 0.0, -1.0   | 0.5    | 0.8, 0.5, 0.5 | Metal      | 0.1  |
| 4        | 0.9, 0.0, -1.0    | 0.5    | 0.8, 0.6, 0.2 | Metal      | 0.5  |
| 5        | 0.0, -0.3, -1.0   | 0.2    | 0.8, 0.8, 0.8 | Metal      | 0.0  |

### Results

| Device            | Time   | Speed up from previous |
|-------------------|--------|------------------------|
| Ray-Caster        | N/A    | N/A                    |
| Ray-Tracer-Legacy | ~122s  | -                      |
| OpenCL-Ray-Tracer | ~0.38s | 321.05x                |
| CUDA-Ray-Tracer   | ~0.25s | 1.52x                  |

The Ray-Caster project doesn't qualify for this benchmark as mainly, it is not an actual raytracer, and it does not have the capability to render materials such as metal and glass.

The huge jump in performance from Ray-Tracer-Legacy and OpenCL-Ray-Tracer is due to the move from single threaded CPU execution to highly parallel GPU kernels.

A decent performance benefit is still found when moving from OpenCL to CUDA due to its more specialised nature and compiler optimisations from NVCC, while essentially just being a port/translation of the OpenCL code.

### Output image

![output](https://user-images.githubusercontent.com/39223201/212554754-de0f2e15-93e3-49d4-ac89-50cbbbfc367e.png)

## To-do

- Reach feature parity with OpenCL-Ray-Tracer
- Figure out where I go from there

## Building the project

Todo (but basically just run the makefile after installing some packages)

## References & Inspiration

- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) by Peter Shirley
- [Coding Adventure: Ray Tracing](https://www.youtube.com/watch?v=Qz0KTGYJtUk) by Sebastian Lague
