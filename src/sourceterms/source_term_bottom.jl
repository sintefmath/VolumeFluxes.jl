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




function evaluate_directional_source_term!(::SourceTermBottom, output, current_state, cs::ConservedSystem, dir::Direction)

    # {right, left}_buffer is (h, hu)
    # output and current_state is (w, hu)
    dx = compute_dx(cs.grid, dir)
    output_momentum = (dir == XDIR) ? output.hu : output.hv
    B = cs.equation.B 
    g = cs.equation.g
    h_right = cs.right_buffer.h
    h_left  = cs.left_buffer.h
    @fvmloop for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
        B_right = B_face_right( B, imiddle, dir)
        B_left  = B_face_left(B, imiddle, dir)

        output_momentum[imiddle] +=-g*((B_right - B_left)/dx)*((h_right[imiddle] + h_left[imiddle])/2.0)
        nothing
    end
end


function evaluate_source_term_twolayer!(::SourceTermBottom, output, current_state, cs::ConservedSystem, dir::Direction)
    dx = compute_dx(cs.grid, dir)
    B  = cs.equation.B
    g  = cs.equation.g
    r  = cs.equation.ρ1 / cs.equation.ρ2   # density ratio

    w_right  = cs.right_buffer.w    
    w_left   = cs.left_buffer.w
    h1_right = cs.right_buffer.h1
    h1_left  = cs.left_buffer.h1

    out_q2 = output.h2u2  # momentum of layer 2
    @fvmloop for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
        B_right = B_face_right(B, imiddle, dir)  # B_{j+1/2}
        B_left  = B_face_left(B, imiddle, dir)   # B_{j-1/2}
        Bx = (B_right - B_left) / dx
        avg = 0.5 * ( w_right[imiddle] + w_left[imiddle] + r*h1_right[imiddle] + r*h1_left[imiddle] )

        out_q2[imiddle] += -g * avg * Bx
        nothing
    end

    return nothing
end
