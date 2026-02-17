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

import CUDA
import Metal
using StaticArrays

# Helper function to convert Float64 to Float32 for nested structures
# Note: Method dispatch order is important here - more specific types first
convert_to_float32(x::SVector{N, Float64}) where N = convert(SVector{N, Float32}, x)
convert_to_float32(x::SVector{N, T}) where {N, T} = x  # Already not Float64, keep as is
convert_to_float32(x::Array{Float64}) = Float32.(x)  # Convert Float64 arrays to Float32  
convert_to_float32(x::Array{Float64, N}) where N = Float32.(x)  # Multi-dimensional Float64 arrays
convert_to_float32(x::AbstractArray) = map(convert_to_float32, x)  # Recursively handle nested arrays
convert_to_float32(x::Float64) = Float32(x)
convert_to_float32(x) = x  # For other types, keep as is

convert_to_backend(backend, array::AbstractArray) = array
convert_to_backend(backend::CUDABackend, array::AbstractArray) = CUDA.CuArray(array)

# Metal backend: convert Float64 to Float32 if backend uses Float32
function convert_to_backend(backend::MetalBackend, array::AbstractArray)
    if backend.realtype == Float32
        # Convert Float64 elements to Float32
        converted_array = convert_to_float32(array)
        Metal.MtlArray(converted_array)
    else
        Metal.MtlArray(array)
    end
end

convert_to_backend(backend::CPUBackend, array::CUDA.CuArray) = collect(array)
convert_to_backend(backend::CPUBackend, array::Metal.MtlArray) = collect(array)

# TODO: Do one for KA?

function create_buffer(backend, number_of_variables::Int64, spatial_resolution)
    zeros(backend.realtype, spatial_resolution..., number_of_variables)
end

function create_buffer(backend::CPUBackend, number_of_variables::Int64, spatial_resolution)
    # TODO: Fixme
    # buffer = KernelAbstractions.zeros(backend.backend, prod(spatial_resolution), number_of_variables)
    buffer = zeros(backend.realtype, spatial_resolution..., number_of_variables)
    buffer
end

function create_buffer(backend::CUDABackend, number_of_variables::Int64, spatial_resolution)
    CUDA.CuArray(zeros(backend.realtype, spatial_resolution..., number_of_variables))
end

function create_buffer(backend::MetalBackend, number_of_variables::Int64, spatial_resolution)
    Metal.MtlArray(zeros(backend.realtype, spatial_resolution..., number_of_variables))
end
