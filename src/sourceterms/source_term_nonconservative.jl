# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


struct SourceTermTwoLayer end

function evaluate_directional_source_term!(::SourceTermTwoLayer, output, current_state, cs::ConservedSystem, dir::Direction)

    dx = compute_dx(cs.grid, dir)
    eq = cs.equation

    g = eq.g
    r = eq.ρ1 / eq.ρ2

    # Update components (2) and (4) of U: q1 (h1u1) and q2 (h2u2).
    out_q1 = output.q1
    out_q2 = output.q2

    @fvmloop for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
        UR = cs.right_buffer[imiddle]  # (h1, q1, w, q2) at x_{j+1/2}^-
        UL = cs.left_buffer[imiddle]   # (h1, q1, w, q2) at x_{j-1/2}^+

        h1R = UR[1]; wR  = UR[3]; h1L = UL[1]; wL  = UL[3]
        N2 = g * 0.5 * ((h1R + wR) + (h1L + wL)) * ((h1R - h1L)/dx)
        N4 = -r * N2
        out_q1[imiddle] += N2
        out_q2[imiddle] += N4

        nothing
    end

    return nothing
end
