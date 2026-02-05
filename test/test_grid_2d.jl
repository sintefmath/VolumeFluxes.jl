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
using StaticArrays
 # We do not technically need this loop, but adding it to create a scope
for _ in get_available_backends()
    nx = 10
    ny = 6
    gc = 2

    grid = VolumeFluxes.CartesianGrid(nx, ny, extent=[0 nx; 0 ny], gc=gc)
    dx = VolumeFluxes.compute_dx(grid)
    dy = VolumeFluxes.compute_dy(grid)

    x_faces = collect(0:nx)
    y_faces = collect(0:ny)
    x_cellcenters = collect(0.5:1:9.5)
    y_cellcenters = collect(0.5:1:5.5)

    faces_from_grid = VolumeFluxes.cell_faces(grid)
    centers_from_grid = VolumeFluxes.cell_centers(grid)
    for j in 1:(ny+1)
        for i in 1:(nx+1)
            @test faces_from_grid[i, j][1] == x_faces[i]
            @test faces_from_grid[i, j][2] == y_faces[j]
        end
    end
    for j in 1:(ny)
        for i in 1:(nx)
            @test centers_from_grid[i, j][1] == x_cellcenters[i]
            @test centers_from_grid[i, j][2] == y_cellcenters[j]
        end
    end

    @test x_faces == VolumeFluxes.cell_faces(grid, XDIR)
    @test y_faces == VolumeFluxes.cell_faces(grid, YDIR)
    @test x_cellcenters == VolumeFluxes.cell_centers(grid, XDIR)
    @test y_cellcenters == VolumeFluxes.cell_centers(grid, YDIR)

    faces_gc = VolumeFluxes.cell_faces(grid, interior=false)
    @test faces_gc[3:end-2, 3:end-2] ≈ faces_from_grid atol = 10^-14 # Got machine epsilon round-off errors
    for j in 1:2
        for i in 1:2
            @test faces_gc[i, j] ≈ SVector{2, Float64}(-3+i, -3+j) atol = 10^-14
            @test faces_gc[end-(i-1), end-(j-1)] ≈ SVector{2, Float64}(nx+(gc-i +1), ny+ (gc-j+1)) atol = 10^-14
        end
    end

    cells_gc = VolumeFluxes.cell_centers(grid, interior=false)
    @test cells_gc[3:end-2, 3:end-2] ≈ centers_from_grid atol = 10^-14
    for j in 1:2
        for i in 1:2
            @test cells_gc[i, j] ≈ SVector{2, Float64}(-3+i + dx/2, -3+j + dy/2)  atol = 10^-14
            @test cells_gc[end-(i-1), end-(j-1)] ≈ SVector{2, Float64}(nx+(gc-i) + dx/2, ny+ (gc-j)+dy/2) atol = 10^-14
        end
    end

end
