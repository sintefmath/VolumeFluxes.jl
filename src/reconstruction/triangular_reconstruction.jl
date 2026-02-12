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

"""
    reconstruct_triangular(grid, cell_values)

Compute limited piecewise-linear gradients on a `TriangularGrid`.

For every cell *i* with centroid ``\\mathbf{c}_i`` and value ``\\bar U_i``,
the three candidate gradients are obtained from the planes through
``\\mathbf{c}_i`` and two of its neighbour centroids:

```math
\\vec{g}_k = \\text{(finite difference through two centroids)}.
```

The final gradient is assembled component-wise with the three-argument
minmod limiter

```math
g^{(d)} = \\operatorname{minmod}(g_1^{(d)}, g_2^{(d)}, g_3^{(d)}).
```

Returns `gradients::Vector{SVector{N,SVector{2,Float64}}}` where `N` is
the number of conserved variables, giving the limited gradient for each
variable in each cell.
"""
function reconstruct_triangular(grid::TriangularGrid, cell_values::AbstractVector{SVector{N,T}}) where {N,T}
    ncells = number_of_cells(grid)
    # Each element stores per-variable 2D gradient
    gradients = Vector{SVector{N,SVector{2,T}}}(undef, ncells)

    zero_grad = SVector{2,T}(zero(T), zero(T))

    for i in 1:ncells
        ci = grid.centroids[i]

        # Collect candidate gradients from each neighbour pair
        # We build one candidate gradient per neighbour edge
        candidate_gx = MVector{3,T}(zero(T), zero(T), zero(T))
        candidate_gy = MVector{3,T}(zero(T), zero(T), zero(T))
        has_neighbour = MVector{3,Bool}(false, false, false)

        for k in 1:3
            nb = grid.neighbors[i][k]
            if nb == 0
                continue
            end
            has_neighbour[k] = true
            cj = grid.centroids[nb]
            dU = cell_values[nb] - cell_values[i]
            dc = cj - ci
            dist2 = dc[1]^2 + dc[2]^2
            # For a single-variable gradient we'd do  g = dU / dist * (dc/dist)
            # but we store per-component below and combine per variable
            # We compute the candidate gradient projection:  g = dU * dc / |dc|^2
            # This gives a gradient such that g · dc = dU.
            # We store only the first component here; we process per variable in the outer loop below.
            for v in 1:N
                # This is a rank-1 approximation; we use it per edge direction
                nothing  # placeholder – we'll compute below
            end
        end

        # For each conserved variable, compute the limited gradient using
        # all three candidate gradients independently.
        var_grads = MVector{N, SVector{2,T}}(ntuple(_ -> zero_grad, Val(N)))

        for v in 1:N
            # Collect up to 3 candidate gradient vectors for variable v
            cand_gx = MVector{3,T}(zero(T), zero(T), zero(T))
            cand_gy = MVector{3,T}(zero(T), zero(T), zero(T))
            n_cands = 0
            for k in 1:3
                nb = grid.neighbors[i][k]
                if nb == 0
                    continue
                end
                n_cands += 1
                cj = grid.centroids[nb]
                dc = cj - ci
                dist2 = dc[1]^2 + dc[2]^2
                dU_v = cell_values[nb][v] - cell_values[i][v]
                # Candidate gradient: g_k = dU_v / |dc|^2 * dc
                cand_gx[n_cands] = dU_v * dc[1] / dist2
                cand_gy[n_cands] = dU_v * dc[2] / dist2
            end

            if n_cands >= 3
                gx = minmod(cand_gx[1], cand_gx[2], cand_gx[3])
                gy = minmod(cand_gy[1], cand_gy[2], cand_gy[3])
            elseif n_cands == 2
                gx = minmod(cand_gx[1], cand_gx[2], zero(T))
                gy = minmod(cand_gy[1], cand_gy[2], zero(T))
            elseif n_cands == 1
                gx = zero(T)
                gy = zero(T)
            else
                gx = zero(T)
                gy = zero(T)
            end
            var_grads[v] = SVector{2,T}(gx, gy)
        end

        gradients[i] = SVector{N,SVector{2,T}}(var_grads)
    end

    return gradients
end

"""
    evaluate_reconstruction(cell_value, gradient, centroid, point)

Evaluate the linearly reconstructed value at `point` given the cell
average `cell_value`, its limited `gradient`, and the cell `centroid`.
"""
function evaluate_reconstruction(cell_value::SVector{N,T}, gradient::SVector{N,SVector{2,T}},
                                 centroid::SVector{2,Float64}, point::SVector{2,Float64}) where {N,T}
    dx = point - centroid
    return SVector{N,T}(ntuple(v -> cell_value[v] + gradient[v][1] * dx[1] + gradient[v][2] * dx[2], Val(N)))
end
