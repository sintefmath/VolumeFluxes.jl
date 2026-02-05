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

macro generate_static_vector_functions(max_dimension)
    expression_to_return = Expr[]
    for dimension in 1:max_dimension
        argument_to_vector_creation = [:(data[index, $i]) for i in 1:dimension]
        vector_creation = :(SVector{$(dimension),RealType}($(argument_to_vector_creation...)))
        function_definition = :(@inline extract_vector(::Val{$(dimension)}, data::AbstractArray{RealType}, index) where {RealType} = $(vector_creation))
        push!(expression_to_return, function_definition)

    end
    return esc(quote
        $(expression_to_return...)
    end)
end
@generate_static_vector_functions 10

@inline function set_vector!(::Val{n}, data, value, index) where {n}
    for i in 1:n
        data[index, i] = value[i]
    end
end
