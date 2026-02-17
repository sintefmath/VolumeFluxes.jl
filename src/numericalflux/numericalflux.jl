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

function compute_flux!(backend, F::NumericalFlux, output, left, right, wavespeeds, grid, equation::Equation, direction)
    Δx = compute_dx(grid, direction)

    @fvmloop for_each_inner_cell(backend, grid, direction) do ileft, imiddle, iright
        F_right, speed_right = F(right[imiddle], left[iright], direction)
        F_left, speed_left = F(right[ileft], left[imiddle], direction)
        output[imiddle] -= 1 / Δx * (F_right - F_left)
        wavespeeds[imiddle] = max(speed_right, speed_left)
        nothing
    end

    return maximum(wavespeeds)
end

"""
    compute_flux!(backend, F, output, state, state, wavespeeds, grid::TriangularGrid, equation, direction)

Specialised `compute_flux!` for triangular grids.  The `direction` argument is
ignored because all edges are processed in a single pass.  Reconstruction
gradients are computed internally via `reconstruct_triangular`, and the
central-upwind numerical flux is evaluated on every edge with boundary
conditions applied directly (no ghost cells).

The `left` and `right` arguments (which on Cartesian grids carry reconstructed
face values) are unused; the current cell-averaged state is read from `left`
(which equals `state` in the standard calling convention).
"""
function compute_flux!(backend, F::NumericalFlux, output, left, right,
                       wavespeeds, grid::TriangularGrid,
                       equation::ShallowWaterEquationsPure, direction)
    ncells = number_of_cells(grid)

    # Extract cell-averaged values from the Volume
    cell_values = Vector{SVector{3, Float64}}(undef, ncells)
    for i in 1:ncells
        cell_values[i] = left[i]
    end

    # Compute limited reconstruction gradients
    gradients = reconstruct_triangular(grid, cell_values)

    # Zero-initialise an accumulator for the RHS contributions
    rhs = zeros(SVector{3, Float64}, ncells)

    # Call the existing triangular flux accumulator
    max_speed = compute_triangular_fluxes!(rhs, grid, equation, cell_values, gradients)

    # Write results into the output Volume and wavespeeds
    @fvmloop for_each_cell(backend, grid) do i
        output[i] += rhs[i]
        wavespeeds[i] = zero(eltype(wavespeeds))
    end

    # Set per-cell wavespeeds by computing max edge speed per cell
    for i in 1:ncells
        cell_speed = zero(Float64)
        ci = grid.centroids[i]
        for k in 1:3
            normal = grid.edge_normals[i][k]
            nb = grid.neighbors[i][k]

            vi1 = grid.triangles[i][k]
            vi2 = grid.triangles[i][k % 3 + 1]
            edge_mid = (grid.nodes[vi1] + grid.nodes[vi2]) / 2.0

            U_minus = evaluate_reconstruction(cell_values[i], gradients[i], ci, edge_mid)

            if nb != 0
                cj = grid.centroids[nb]
                U_plus = evaluate_reconstruction(cell_values[nb], gradients[nb], cj, edge_mid)
            else
                U_plus = _boundary_state(grid.boundary, U_minus, normal)
            end

            _, speed = central_upwind_flux(equation, U_minus, U_plus, normal)
            cell_speed = max(cell_speed, speed)
        end
        wavespeeds[i] = cell_speed
    end

    return max_speed
end


include("swe/centralupwind.jl")
include("burgers/godunov.jl")
include("burgers/rusanov.jl")
include("triangular_flux.jl")
