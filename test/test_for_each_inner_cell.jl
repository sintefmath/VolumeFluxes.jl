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
using CUDA
using Test

for backend in get_available_backends()
    nx = 10
    grid = VolumeFluxes.CartesianGrid(nx)
    backend = make_cpu_backend()

    leftarrays = 1000 * ones(nx + 2)
    middlearrays = 1000 * ones(nx + 2)
    rightarrays = 1000 * ones(nx + 2)

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_inner_cell(backend, grid, XDIR) do ileft, imiddle, iright
        leftarrays[imiddle] = ileft
        middlearrays[imiddle] = imiddle
        rightarrays[imiddle] = iright
    end

    @test leftarrays[1] == 1000
    @test middlearrays[1] == 1000
    @test rightarrays[1] == 1000


    @test leftarrays[end] == 1000
    @test middlearrays[end] == 1000
    @test rightarrays[end] == 1000

    @test leftarrays[2:end-1] == 1:(nx)
    @test middlearrays[2:end-1] == 2:(nx+1)
    @test rightarrays[2:end-1] == 3:(nx+2)


    ## Check for ghost cells


    leftarrays = 1000 * ones(nx + 2)
    middlearrays = 1000 * ones(nx + 2)
    rightarrays = 1000 * ones(nx + 2)

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_inner_cell(backend, grid, XDIR; ghostcells=3) do ileft, imiddle, iright
        leftarrays[imiddle] = ileft
        middlearrays[imiddle] = imiddle
        rightarrays[imiddle] = iright
    end

    for i in 1:3
        @test leftarrays[i] == 1000
        @test middlearrays[i] == 1000
        @test rightarrays[i] == 1000


        @test leftarrays[end-i+1] == 1000
        @test middlearrays[end-i+1] == 1000
        @test rightarrays[end-i+1] == 1000
    end
    @test leftarrays[4:end-3] == 3:(nx-2)
    @test middlearrays[4:end-3] == 4:(nx-1)
    @test rightarrays[4:end-3] == 5:(nx)
end
