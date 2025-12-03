# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# ## Example
# We create a simple simulation of the Advection equation with different limiters
using SinFVM
using CairoMakie
using StaticArrays

#Add in initial condition helper and analytical solution
# Classic advection test IC (bump + step)
function classic_advection_IC(x)
    if 0.1 ≤ x && x ≤ 0.4
        # Smooth cosine bump centered at 0.25 with width 0.3
        return 0.5 * (1 + cos(pi * (x - 0.25) / 0.15))
    elseif 0.5 ≤ x && x ≤ 0.9
        # Step
        return 1.0
    else
        return 0.0
    end
end
# Exact periodic solution for linear advection u_t + a u_x = 0
function analytic_advection(x::AbstractVector, T; a::Real = 1.0, ic::Function = classic_advection_IC, L::Real = 1.0)
    u_exact = similar(x)
    @inbounds for i in eachindex(x)
        ξ = x[i] - a*T
        # wrap into [0, L)
        ξ -= floor(ξ / L) * L
        u_exact[i] = ic(ξ)
    end
    return u_exact
end


#Make a runner for different choices of limiters
function run_advection_with_limiter(N::Int, L::Real, T::Real; a::Real = 1.0, CFL::Float64 = 0.6, limiter_obj)
    backend = make_cpu_backend()
    # 2 ghost cells, periodic BC
    grid = CartesianGrid(N; gc = 2, boundary = SinFVM.PeriodicBC())

    equation       = Advection(a)
    reconstruction = LinearLimiterReconstruction(limiter_obj)
    timestepper    = RungeKutta2()
    numericalflux  = Godunov(equation)
    system         = ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    simulator      = Simulator(backend, system, timestepper, grid)
    # Interior cell centers (no ghosts)
    x = SinFVM.cell_centers(grid)
    # Initial condition packed as SVector (single conserved variable)
    u0 = x -> @SVector[ classic_advection_IC(x) ]
    initial = u0.(x)
    SinFVM.set_current_state!(simulator, initial)

    # Simple CFL-based max timestep
    dx = L / N
    dtmax = CFL * dx / abs(a)
    simulate_to_time(simulator, T; maximum_timestep = dtmax)

    # Extract scalar u from Volume of SVectors
    u_state = SinFVM.current_interior_state(simulator).u
    u_num = [u_state[i][1] for i in eachindex(u_state)]
    return x, u_num
end

# Make function for visualizing results
function sinfvm_advection_compare_limiters(N, L, T; a::Real = 1.0, CFL::Float64 = 0.6,
        saveprefix::Union{Nothing,AbstractString} = nothing)
    limiter_map = Dict( 
        :minmod   => SinFVM.MinmodLimiter(1.0),  # θ = 1 → classical minmod
        :mc       => SinFVM.MCLimiter(),
        :superbee => SinFVM.SuperbeeLimiter(),
        :vanleer  => SinFVM.VanLeerLimiter(),
    )
    results = Dict{Symbol,NamedTuple}()
    for (name, limobj) in limiter_map
        @info "Running SinFVM advection with limiter = $(name)"
        x, u_num = run_advection_with_limiter(N, L, T; a = a, CFL = CFL, limiter_obj = limobj)
        u_exact = analytic_advection(x, T; a = a, ic = classic_advection_IC, L = L)
        fig = Figure()
        ax  = Axis(fig[1,1], title  = "Limiter: $(String(name)), N=$N, T=$T, a=$a", xlabel = "x", ylabel = "u")
        lines!(ax, x, u_exact; label = "exact", color = "#009AFA", linewidth = 1)
        scatter!(ax, x, u_num; label = "numeric", color = "#E36F47", markersize  = 5, marker = :circle, strokecolor = :black, strokewidth = 1)

        axislegend(ax, position = :lt)
        display(fig)
        if saveprefix !== nothing
            fname = "$(saveprefix)_$(String(name)).png"
            try
                savefig(fig, fname)
            catch err
                @warn "Could not save figure" error=err path=fname
            end
        end
        results[name] = (x = collect(x), u = collect(u_num), u_exact = collect(u_exact))
    end
    return results
end


# Run the simulation
N  = 200
L  = 1.0
T  = 1.0
a  = 1.0
CFL = 0.6

results = sinfvm_advection_compare_limiters(N, L, T; a = a, CFL = CFL, saveprefix = nothing)



