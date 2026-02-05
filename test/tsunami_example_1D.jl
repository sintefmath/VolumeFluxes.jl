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
using StaticArrays
using CairoMakie

function tsunami(;T=10, dt=1, w0_height=1.0, bump=false)

    nx = 1024
    grid = VolumeFluxes.CartesianGrid(nx; gc=2, boundary=VolumeFluxes.WallBC(), extent=[0.0  1000.0], )
    function terrain(x)
        b =  25/1000*(x - 800)
        if bump
            b += 8*exp(-(x - 500)^2/100)
        end
        return b
    end
    
    B_data = Float64[terrain(x) for x in VolumeFluxes.cell_faces(grid, interior=false)]

    backend = make_cpu_backend()
    B = VolumeFluxes.BottomTopography1D(B_data, backend, grid)
    eq = VolumeFluxes.ShallowWaterEquations1D(B)
    rec = VolumeFluxes.LinearReconstruction(2)
    flux = VolumeFluxes.CentralUpwind(eq)
    bst = VolumeFluxes.SourceTermBottom()
    conserved_system = VolumeFluxes.ConservedSystem(backend, rec, flux, eq, grid, bst)
    
    #balance_system = VolumeFluxes.BalanceSystem(conserved_system, bst)
    
    
    timestepper = VolumeFluxes.ForwardEulerStepper()
    simulator = VolumeFluxes.Simulator(backend, conserved_system, timestepper, grid, cfl=0.1)
    
    x = VolumeFluxes.cell_centers(grid)
    xf = VolumeFluxes.cell_faces(grid)
    u0 = x -> @SVector[w0_height*(x < 100), 0.0]
    initial = u0.(x)

    VolumeFluxes.set_current_state!(simulator, initial)
    all_t = []
    all_h = []
    all_hu = []
    t = 0.0
    while t < T
        t += dt
        VolumeFluxes.simulate_to_time(simulator, t)
        
        state = VolumeFluxes.current_interior_state(simulator)
        push!(all_h, collect(state.h))
        push!(all_hu, collect(state.hu))
        push!(all_t, t)
    end

    index = Observable(1)
    h = @lift(all_h[$index])
    hu = @lift(all_hu[$index])
    t = @lift(all_t[$index])

    f = Figure(size=(800, 800), fontsize=24)
    infostring = @lift("tsunami, t=$(all_t[$index]) nx=$(nx)") #
$(split(typeof(rec),".")[2]) and $(split(typeof(flux), ".")[2])"
        ax_h = Axis(
            f[1, 1],
            title=infostring,
            ylabel="h",
            xlabel="x",
        )

        lines!(ax_h, x, h, color="blue", label='w')
        lines!(ax_h, xf, collect(B.B[3:end-2]), label='B', color="red")

        ax_u = Axis(
            f[2, 1],
            title="hu",
            ylabel="hu",
            xlabel="x",
        )
        ylims!(ax_u, -3, 8)

        lines!(ax_u, x, hu, label="hu")

        record(f, "tsunami.mp4", 1:length(all_t); framerate=10) do i 
            index[] =i                
        end
        #display(f)
end

tsunami(T=200; bump=true)
println("done")
nothing
