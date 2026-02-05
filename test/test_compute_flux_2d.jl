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

using Test
using SinFVM
using StaticArrays
import CUDA
using LinearAlgebra

function test_compute_flux_2d(backend)
    backend_name = SinFVM.name(backend)
    u0 = x -> @SVector[exp.(-(norm(x .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0]
    nx = 32
    ny = 16
    grid = SinFVM.CartesianGrid(nx, ny; gc=1)

    equation = SinFVM.ShallowWaterEquationsPure()

    reconstruction = SinFVM.NoReconstruction()
    numericalflux = SinFVM.CentralUpwind(equation)

    x = SinFVM.cell_centers(grid)
    initial = u0.(x)

    state = SinFVM.Volume(backend, equation, grid)
    output_state = SinFVM.Volume(backend, equation, grid)
    interior_state = SinFVM.InteriorVolume(state)
    CUDA.@allowscalar interior_state[:, :] = initial
    SinFVM.update_bc!(backend, grid.boundary, grid, equation, state)

    for j in 1:ny
        for i in 1:nx
            equation_computed_x = equation(XDIR, initial[i, j]...)
            @test !any(isnan.(equation_computed_x))
            equation_computed_y = equation(XDIR, initial[i, j]...)
            @test !any(isnan.(equation_computed_y))

            eigenvalues_x = SinFVM.compute_eigenvalues(equation, XDIR, initial[i, j]...)
            @test !any(isnan.(eigenvalues_x))

            eigenvalues_y = SinFVM.compute_eigenvalues(equation, YDIR, initial[i, j]...)
            @test !any(isnan.(eigenvalues_y))
        end
    end
    wavespeeds = SinFVM.create_scalar(backend, grid, equation)
    SinFVM.compute_flux!(backend, numericalflux, output_state, state, state, wavespeeds, grid, equation, XDIR)


    @test !any(isnan.(wavespeeds))
    @test !any(isnan.(collect(output_state.h)))
    @test !any(isnan.(collect(output_state.hv)))
    @test !any(isnan.(collect(output_state.hu)))

    SinFVM.compute_flux!(backend, numericalflux, output_state, state, state, wavespeeds, grid, equation, YDIR)

    @test !any(isnan.(wavespeeds))
    @test !any(isnan.(collect(output_state.h)))
    @test !any(isnan.(collect(output_state.hv)))
    @test !any(isnan.(collect(output_state.hu)))
end

for backend in get_available_backends()
    test_compute_flux_2d(backend)
end
