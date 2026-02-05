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


function update_bc!(backend, ::PeriodicBC, grid::CartesianGrid{1}, ::Equation, data)
    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        data[ghostcell] = data[grid.totalcells[1]+ghostcell-2*grid.ghostcells[1]]
        data[grid.totalcells[1]-(grid.ghostcells[1]-ghostcell)] = data[grid.ghostcells[1]+ghostcell]
    end
end

function update_bc!(backend, ::PeriodicBC, grid::CartesianGrid{2}, ::Equation, data)
    # TODO: Introduce some helper functions here...
    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        data[ghostcell] = data[grid.totalcells[1]+ghostcell[1]-2*grid.ghostcells[1], ghostcell[2]]
        data[grid.totalcells[1]-(grid.ghostcells[1]-ghostcell[1]), ghostcell[2]] = data[grid.ghostcells[1]+ghostcell[1], ghostcell[2]]
    end

    @fvmloop for_each_ghost_cell(backend, grid, YDIR) do ghostcell
        data[ghostcell] = data[ghostcell[1], grid.totalcells[2]+ghostcell[2]-2*grid.ghostcells[2]]
        data[ghostcell[1], grid.totalcells[2]-(grid.ghostcells[2]-ghostcell[2])] = data[ghostcell[1], grid.ghostcells[2]+ghostcell[2]]
    end
end

function update_bc!(backend, ::NeumannBC, grid::CartesianGrid{1}, ::Equation, data)
    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        data[ghostcell] = data[2 * grid.ghostcells[1] - ghostcell + 1]
        data[grid.totalcells[1]-(grid.ghostcells[1]-ghostcell)] = data[grid.totalcells[1] - ghostcell + 1]
    end
end

function update_bc!(backend, ::NeumannBC, grid::CartesianGrid{2}, ::Equation, data)
    # TODO: Introduce some helper functions here...
    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        data[ghostcell[1], ghostcell[2]] = data[2 * grid.ghostcells[1] - ghostcell[1] + 1, ghostcell[2]]
        data[grid.totalcells[1]-(grid.ghostcells[1]-ghostcell[1]), ghostcell[2]] = data[grid.totalcells[1] - ghostcell[1] + 1, ghostcell[2]]
    end

    @fvmloop for_each_ghost_cell(backend, grid, YDIR) do ghostcell
        data[ghostcell[1], ghostcell[2]] = data[ghostcell[1], grid.totalcells[2]+ghostcell[2]-2*grid.ghostcells[2]]
        data[ghostcell[1], grid.totalcells[2]-(grid.ghostcells[2]-ghostcell[2])] = data[ghostcell[1], grid.totalcells[2]  - ghostcell[2] + 1]
    end
end


function update_bc!(backend, ::WallBC, grid::CartesianGrid{1}, ::AllSWE1D, data)
    function local_update_ghostcell!(data, ghost, inner)
        # h(ghost) = h(inner) and hu(ghost) = -hu(inner)
        data[ghost] = typeof(data[ghost])(data[inner][1], -data[inner][2])
    end

    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        left_ghostcell = ghostcell
        left_innercell = grid.ghostcells[1] * 2 + 1 - ghostcell
        local_update_ghostcell!(data, left_ghostcell, left_innercell)

        right_ghostcell = grid.totalcells[1] - grid.ghostcells[1] + ghostcell
        right_innercell = grid.totalcells[1] - grid.ghostcells[1] - ghostcell + 1
        local_update_ghostcell!(data, right_ghostcell, right_innercell)
    end
end


function update_bc!(backend, ::WallBC, grid::CartesianGrid{2}, ::AllSWE2D, data)

    function local_update_ghostcell!(data, ghost, inner, ::XDIRT)
        # h(ghost) = h(inner),  hu(ghost) = -hu(inner), hv(ghost) = hv(inner)
        data[ghost] = typeof(data[ghost])(data[inner][1], -data[inner][2], data[inner][3])
    end
        function local_update_ghostcell!(data, ghost, inner, ::YDIRT)
        # h(ghost) = h(inner),  hu(ghost) = hu(inner), hv(ghost) = -hv(inner)
        data[ghost] = typeof(data[ghost])(data[inner][1], data[inner][2], -data[inner][3])
    end

    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        left_ghostcell = ghostcell
        left_innercell = CartesianIndex(grid.ghostcells[1] * 2 + 1 - ghostcell[1], ghostcell[2])
        local_update_ghostcell!(data, left_ghostcell, left_innercell, XDIR)

        right_ghostcell = CartesianIndex(grid.totalcells[1] - grid.ghostcells[1] + ghostcell[1], ghostcell[2])
        right_innercell = CartesianIndex(grid.totalcells[1] - grid.ghostcells[1] - ghostcell[1] + 1, ghostcell[2])
        local_update_ghostcell!(data, right_ghostcell, right_innercell, XDIR)
    end
    @fvmloop for_each_ghost_cell(backend, grid, YDIR) do ghostcell
        left_ghostcell = ghostcell
        left_innercell = CartesianIndex(ghostcell[1], grid.ghostcells[2] * 2 + 1 - ghostcell[2])
        local_update_ghostcell!(data, left_ghostcell, left_innercell, YDIR)

        right_ghostcell = CartesianIndex(ghostcell[1], grid.totalcells[2] - grid.ghostcells[2] + ghostcell[2])
        right_innercell = CartesianIndex(ghostcell[1], grid.totalcells[2] - grid.ghostcells[2] - ghostcell[2] + 1)
        local_update_ghostcell!(data, right_ghostcell, right_innercell, YDIR)
    end
end


update_bc!(backend, grid::CartesianGrid, eq::Equation, data) = update_bc!(backend, grid.boundary, grid, eq, data)
update_bc!(simulator::Simulator, data) = update_bc!(simulator.backend, simulator.grid, simulator.system.equation, data)
