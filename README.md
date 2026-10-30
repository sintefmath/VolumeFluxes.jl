<p align="center">
  <img src="docs/src/assets/volume_fluxes_small.png" alt="VolumeFluxes.jl" width="480">
</p>

<p align="center">
  <a href="https://github.com/sintefmath/VolumeFluxes.jl/actions/workflows/tests.yml"><img src="https://github.com/sintefmath/VolumeFluxes.jl/actions/workflows/tests.yml/badge.svg?branch=main" alt="CPU tests"></a>
  <a href="https://github.com/sintefmath/VolumeFluxes.jl/actions/workflows/docs.yml"><img src="https://github.com/sintefmath/VolumeFluxes.jl/actions/workflows/docs.yml/badge.svg?branch=main" alt="Documentation build"></a>
  <a href="https://sintefmath.github.io/VolumeFluxes.jl/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT license"></a>
</p>

**VolumeFluxes.jl** is a Julia framework for solving conservation and balance laws with the finite volume method, with an emphasis on modelling surface water using the shallow water equations. It is developed by the [Applied Computational Science](https://www.sintef.no/en/digital/departments-new/department-of-mathematics-and-cybernetics/research-group-applied-computational-science/) research group at [SINTEF Digital](https://www.sintef.no/en/digital/).

## Features

- Shallow water equations in one and two dimensions, with bottom topography, friction, rain and infiltration source terms
- CPU and CUDA GPU backends through KernelAbstractions.jl
- Differentiable simulations with ForwardDiff.jl for optimization and inverse problems
- Composable grids, numerical fluxes, reconstructions, time steppers and callbacks

## Getting started

From the root of the repository, instantiate the project environment:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Then run the test suite to verify the installation:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Examples

Complete simulation scenarios live in [`examples/`](examples):

| Example | Description |
| --- | --- |
| [`shallow_water_1d.jl`](examples/shallow_water_1d.jl) | 1D shallow water equations on a flat bottom |
| [`terrain.jl`](examples/terrain.jl) | Rain and infiltration on a terrain with a bay |
| [`optimization.jl`](examples/optimization.jl) | Wall placement optimized with automatic differentiation |
| [`callbacks.jl`](examples/callbacks.jl) | Monitoring a simulation with callbacks |
| [`urban.jl`](examples/urban.jl) | Flow over an urban terrain model with buildings (requires local data files) |

Most of these are also rendered in the [documentation](https://sintefmath.github.io/VolumeFluxes.jl/).

## Testing

The backend used by the test suite is selected with the `VOLUMEFLUXES_BACKEND` environment variable:

| Value | Behaviour |
| --- | --- |
| `auto` (default) | CPU, plus CUDA when `CUDA.functional()` is true |
| `cpu` | CPU only, without probing for a CUDA driver or device |
| `gpu` | CPU plus CUDA; fails instead of silently falling back |

```bash
VOLUMEFLUXES_BACKEND=cpu julia --project -e 'using Pkg; Pkg.test()'
```

Explicit `make_cuda_backend()` calls still request CUDA regardless of this setting, and the suite logs whether GPU coverage was included or skipped.

Continuous integration runs the CPU test suite on Julia 1.10 and the latest stable 1.x release for every pull request and push to `main`. GPU tests are run manually on a dedicated CUDA runner.

## Documentation

The documentation is published at <https://sintefmath.github.io/VolumeFluxes.jl/>. To build it locally (this takes a few minutes):

```bash
julia --project docs/make.jl
```

Then serve it with [LiveServer.jl](https://github.com/JuliaDocs/LiveServer.jl) and open <http://localhost:8000>:

```bash
julia -e 'using LiveServer; serve(dir="docs/build")'
```

Examples are converted with Literate.jl without being executed, so building the documentation does not require a GPU.

## License

VolumeFluxes.jl is released under the [MIT License](LICENSE). Copyright (c) 2024 SINTEF AS.
