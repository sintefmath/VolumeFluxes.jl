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

struct HortonInfiltration{S,T} <: SourceTermInfiltration
    # Empiric infiltration model, where the infiltration rate f (m/s) is given by
    # f(t) = fc + (f0 - fc)*exp(-k*t)
    # The default parameters are supposed to correspond to a sandy soil type.
    # 
    # Source: 
    # Eq (9), Table  1, and Sections 2.2.1 and 3.1 in:
    # Fernandez-Pato, Caviedes-Voullieme, Garcia-Navarro (2016), 
    # "Rainfall/runoff simulation with  2D full shallow water equations: Sensitivity 
    #  analysis and calibration of infiltration parameters".
    # Journal of Hydrology, 536, 496-513 
    # https://doi.org/10.1016/j.jhydrol.2016.03.021.
    f0::S
    fc::S
    k::S
    factor::T
    function HortonInfiltration(grid::CartesianGrid{2}, backend; factor=1.0, fc=3.272e-5, f0=1.977e-4, k=2.43e-3)
        if ndims(factor) == 0
            factor = [factor for x in cell_centers(grid; interior=false)]
        else
            validate_infiltration_factor(factor, grid)
        end
        factor = convert_to_backend(backend, factor)
        return new{typeof(f0), typeof(factor)}(f0, fc, k, factor)
    end
    HortonInfiltration(f0, fc, k, factor; should_never_be_called) = new{typeof(f0), typeof(factor)}(f0, fc, k, factor)
end

function Adapt.adapt_structure(
    to,
    infiltration::HortonInfiltration
) 
    f0 = Adapt.adapt_structure(to, infiltration.f0)
    fc = Adapt.adapt_structure(to, infiltration.fc)
    k  = Adapt.adapt_structure(to, infiltration.k)
    factor = Adapt.adapt_structure(to, infiltration.factor)  
    HortonInfiltration(f0, fc, k, factor; should_never_be_called=nothing)
end


function validate_infiltration_factor(factor, grid::Grid)
    if size(factor) != size(grid) 
        throw(DomainError("Infiltration factor should be of size $(size(grid)) but got $(size(factor))"))
    end
end


function compute_infiltration(f::HortonInfiltration, t, index::CartesianIndex)
    return f.factor[index]*(f.fc + (f.f0 - f.fc)*exp(-f.k*t))
end


struct ConstantInfiltration{S} <: SourceTermInfiltration
    infiltration_rate::S
end



function compute_infiltration(f::ConstantInfiltration, t, index::CartesianIndex)
    return f.infiltration_rate
end

function evaluate_source_term!(infiltration::SourceTermInfiltration, output, current_state, cs::ConservedSystem, t)
    output_h = output.h
    @fvmloop for_each_cell(cs.backend, cs.grid) do index
        output_h[index] -= compute_infiltration(infiltration, t, index)
    end
end
