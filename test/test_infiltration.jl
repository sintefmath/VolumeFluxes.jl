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

using StaticArrays
using LinearAlgebra
using Test
import CUDA
using Logging
using VolumeFluxes


function plain_infiltration(backend, infiltration, grid, T; t0=0.0)
    equation = VolumeFluxes.ShallowWaterEquations()
    reconstruction = VolumeFluxes.LinearReconstruction()
    numericalflux = VolumeFluxes.CentralUpwind(equation)

    conserved_system =
        VolumeFluxes.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, infiltration)
    timestepper = VolumeFluxes.ForwardEulerStepper()
    simulator = VolumeFluxes.Simulator(backend, conserved_system, timestepper, grid; t0=t0)

    u0 = x -> @SVector[1.0, 0.0, 0.0]
    x = VolumeFluxes.cell_centers(grid)
    initial = u0.(x)
    VolumeFluxes.set_current_state!(simulator, initial)

    # TODO: Make callback function that obtains the accumulated infiltration and runoff

    @time VolumeFluxes.simulate_to_time(simulator, t0 + T)

    results = VolumeFluxes.current_interior_state(simulator)
    @test all(collect(results.hu) .== 0.0)
    @test all(collect(results.hv) .== 0.0)
    return collect(results.h)
end

function test_infiltration()
    grid = VolumeFluxes.CartesianGrid(10, 10; gc=2, extent=[0 100; 0 100])
    backend = VolumeFluxes.make_cpu_backend()
    test_inf = VolumeFluxes.HortonInfiltration(grid, backend)
    @test all(test_inf.factor .== 1.0)
    @test size(test_inf.factor) ==  size(grid)
    @test VolumeFluxes.compute_infiltration(test_inf, 0.0, CartesianIndex(1, 1)) == test_inf.f0
    @test VolumeFluxes.compute_infiltration(test_inf, 1e6, CartesianIndex(1, 1)) == test_inf.fc

    factor_bad_size = [1.0 for x in VolumeFluxes.cell_centers(grid)]
    @test_throws DomainError VolumeFluxes.HortonInfiltration(VolumeFluxes.CartesianGrid(2,2), backend; factor=factor_bad_size)

    if VolumeFluxes.has_cuda_backend()
        cuda_backend = VolumeFluxes.make_cuda_backend()
        infiltration_cuda = VolumeFluxes.HortonInfiltration(grid, cuda_backend)

        h_cpu = plain_infiltration(backend, test_inf, grid, 1000; t0=1e6)
        @test maximum(h_cpu) .≈ ( 1.0 - test_inf.fc*1000) atol=1e-10
        @test minimum(h_cpu) .≈ ( 1.0 - test_inf.fc*1000) atol=1e-10

        h_cuda = plain_infiltration(cuda_backend, infiltration_cuda, grid, 1000; t0=1e6)
        @test maximum(h_cuda) .≈ ( 1.0 - infiltration_cuda.fc*1000) atol=1e-10
        @test minimum(h_cuda) .≈ ( 1.0 - infiltration_cuda.fc*1000) atol=1e-10
    else
        @debug "CUDA not available, skipping CUDA tests"
    end
    # grid_case1 = VolumeFluxes.CartesianGrid(2000, 10;  gc=2, boundary=VolumeFluxes.WallBC(), extent=[0.0 4000.0; 0.0 20.0])
end

test_infiltration()
