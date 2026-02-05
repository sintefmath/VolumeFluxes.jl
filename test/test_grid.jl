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

using VolumeFluxes
using Test
 # We do not technically need this loop, but adding it to create a scope
for _ in get_available_backends()
    nx = 10
    grid = VolumeFluxes.CartesianGrid(nx, extent=[0 nx], gc=2)

    x_faces = collect(0:nx)
    x_cellcenters = collect(0.5:1:9.5)

    @test VolumeFluxes.cell_faces(grid) == x_faces
    @test VolumeFluxes.cell_centers(grid) == x_cellcenters

    const_b = 3.14
    const_B = VolumeFluxes.constant_bottom_topography(grid, const_b)

    @test size(const_B)[1] == nx + 4 + 1
    for b in const_B
        @test b == const_b
    end

    x_faces_gc = VolumeFluxes.cell_faces(grid, interior=false)
    @test x_faces_gc[3:end-2] ≈ x_faces atol = 10^-14 # Got machine epsilon round-off errors
    @test x_faces_gc[1:2] ≈ [-2, -1] atol = 10^-14
    @test x_faces_gc[end-1:end] ≈ [11, 12] atol = 10^-14

    x_cells_gc = VolumeFluxes.cell_centers(grid, interior=false)
    @test x_cells_gc[3:end-2] ≈ x_cellcenters atol = 10^-14
    @test x_cells_gc[1:2] ≈ [-1.5, -0.5] atol = 10^-14
    @test x_cells_gc[end-1:end] ≈ [10.5, 11.5] atol = 10^-14
end
