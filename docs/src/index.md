```@raw html
<div class="vf-hero">
```

![VolumeFluxes.jl logo](assets/volume_fluxes_small.png)

```@raw html
<p class="vf-tagline">Finite volume solvers for conservation and balance laws in Julia,
with an emphasis on modelling surface water with the shallow water equations.</p>
<div class="vf-buttons">
```

[Get started](@ref "Getting started") [Browse the examples](terrain.md)

```@raw html
</div>
</div>
```

# What is VolumeFluxes.jl?

VolumeFluxes (SINTEF Finite Volume Methods) is a Julia framework for solving conservation and balance laws using the finite volume method, with a strong emphasis on modelling of surface water. It is developed and maintained by the
[Applied Computational Science](https://www.sintef.no/en/digital/departments-new/department-of-mathematics-and-cybernetics/research-group-applied-computational-science/)
research group at [SINTEF Digital](https://www.sintef.no/en/digital/).

```@raw html
<div class="vf-cards">
```

- **Shallow water equations** Solvers in one and two dimensions, with bottom topography, friction, rain and infiltration source terms.
- **CPU and GPU backends** Built on KernelAbstractions.jl, so the same model runs on a laptop or on a CUDA device.
- **Differentiable simulations** Simulations work with ForwardDiff.jl, enabling gradient-based optimization and inverse problems.
- **Extensible building blocks** Grids, numerical fluxes, reconstructions, time steppers and callbacks that can be combined freely.

```@raw html
</div>
```

## Getting started

From the root of the repository, instantiate the project environment:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

A minimal one-dimensional shallow water simulation sets up a backend, a grid, an equation and a simulator:

```julia
using VolumeFluxes, StaticArrays

backend = make_cpu_backend()
grid = CartesianGrid(1024; gc=2)
```

The [1D shallow water example](shallow_water_1d.md) walks through the remaining steps in detail.

## Examples

```@raw html
<div class="vf-cards">
```

- **[Shallow water in 1D](shallow_water_1d.md)** A Gaussian bump of water on a flat bottom — the best place to start.
- **[Terrain with a bay](terrain.md)** Rain and Horton infiltration on a real terrain, modelled with the 2D shallow water equations.
- **[Optimizing wall placement](optimization.md)** Use automatic differentiation to place a wall that protects against a dam break.
- **[Callbacks](callbacks.md)** Monitor and post-process the solution while the simulation runs, without storing every time step.

```@raw html
</div>
```

## Mathematical foundation

We consider a general (hyperbolic) conservation or balance law

$$u_t + \nabla_x \cdot F(u) = S(u)\qquad \text{on}\qquad D\times [0, T)$$

where $u:D\times[0,T)\to\mathbb{R}$ is a vector of conserved (balanced) quantities, $F$ is a flux function and $S$ represents source terms. It is common to discretize such equations with the finite volume method.
