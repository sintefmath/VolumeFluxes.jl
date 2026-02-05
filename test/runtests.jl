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

# TODO: Go through tests and check they do not take longer time than necessary
using CairoMakie

# Disable showing the plot in CairoMakie
CairoMakie.activate!(type = "svg")
@testset "VolumeFluxes tests" begin
    # Run all scripts in test/test_*.jl
    ls_test = readdir("test")
    for test_file in readdir("test")
        if startswith(test_file, "test_") && endswith(test_file, ".jl")
            #@show test_name, test_file
            
            test_name = replace(test_file, ".jl"=>"")
            @testset "$(test_name)" begin
                include(test_file)
            end
      
        end
    end
end
nothing
