# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using Cthulhu
using StaticArrays
using Test
using CairoMakie
module Correct
include("fasit.jl")
end
using VolumeFluxes
function run_simulation()
    u0 = x -> sin.(2π * x) .+ 1.5
    nx = 16 * 1024
    grid = VolumeFluxes.CartesianGrid(nx)
    backend = make_cpu_backend()

    equation = VolumeFluxes.Burgers()
    reconstruction = VolumeFluxes.NoReconstruction()
    numericalflux = VolumeFluxes.Godunov(equation)
    conserved_system = VolumeFluxes.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = VolumeFluxes.ForwardEulerStepper()

    x = VolumeFluxes.cell_centers(grid)
    initial = collect(map(z -> SVector{1,Float64}([z]), u0(x)))
    simulator = VolumeFluxes.Simulator(backend, conserved_system, timestepper, grid)

    VolumeFluxes.set_current_state!(simulator, initial)
    current_state = VolumeFluxes.current_state(simulator)
    @test current_state[1] == current_state[end-1]
    @test current_state[end] == current_state[2]
    t = 0.0

    T = 0.7



    swe_timesteps = 0
    count_timesteps(varargs...) = swe_timesteps += 1

    @time VolumeFluxes.simulate_to_time(simulator, T, callback=count_timesteps)
    @show swe_timesteps
    VolumeFluxes.set_current_state!(simulator, initial)
    @time VolumeFluxes.simulate_to_time(simulator, T)
    # @profview VolumeFluxes.simulate_to_time(simulator, T)
    f = Figure(size=(1600, 600), fontsize=24)

    ax = Axis(f[1, 1], title="Comparison",
        ylabel="Solution",
        xlabel="x",
    )


    lines!(ax, x, first.(initial), label=L"u_0(x)")
    result = collect(VolumeFluxes.current_interior_state(simulator).u)
    lines!(ax, x, result, linestyle=:dot, color=:red, linewidth=7, label=L"u^{\Delta x}(x, t)")

    number_of_x_cells = nx

    println("Running bare bones twice")
    @time xcorrect, ucorrect, _ = Correct.solve_fvm(u0, T, number_of_x_cells, Correct.Burgers())
    @time xcorrect, ucorrect, timesteps = Correct.solve_fvm(u0, T, number_of_x_cells, Correct.Burgers())
    @show timesteps
    lines!(ax, xcorrect, ucorrect, label="Reference solution")


    axislegend(ax)

    display(f)

end

run_simulation()
