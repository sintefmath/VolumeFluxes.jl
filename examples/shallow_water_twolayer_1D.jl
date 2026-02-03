using CairoMakie
using Cthulhu
using StaticArrays
using SinFVM

#Copy of the shallow water 1D example, but for the two-layer shallow water equations
backend = make_cpu_backend()
nx = 1024
grid = CartesianGrid(nx; gc=2)
u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .+ 1.5, 0.0 .* x]
x = SinFVM.cell_centers(grid)
initial = u0.(x)


equation = SinFVM.TwoLayerShallowWaterEquations1D() #Use defaults for now: ρ1=1, ρ2=1.02, g=9.81, 

# We choose a linear reconstruction with the Van Leer limiter
reconstruction = LinearLimiterReconstruction(SinFVM.VanLeerLimiter())

# We choose the Central Upwind numerical flux
numericalflux = CentralUpwind(equation)

# And we setup the conserved system
conserved_system = ConservedSystem(backend, reconstruction, numericalflux, equation, grid)



# Note that we use the RungeKutta2 method to get second order in time as well
timestepper = RungeKutta2()

# We define the final time
T = 0.05
# Then we create the simulator
simulator = Simulator(backend, conserved_system, timestepper, grid)

# Set the initial data created above
SinFVM.set_current_state!(simulator, initial)

# --- Visualization setup ---
f = Figure(size=(1600, 600), fontsize=24)

ax_h = Axis(
    f[1, 1],
    title="Two-layer SWE 1D. Central-Upwind + RungeKutta2. nx=$(nx), T=$(T)",
    ylabel="layer depths",
    xlabel=L"x",
)

ax_q = Axis(
    f[1, 2],
    title="Two-layer SWE 1D. Central-Upwind + RungeKutta2. nx=$(nx), T=$(T)",
    ylabel="layer momenta",
    xlabel=L"x",
)

# --- Initial state ---
initial_state = SinFVM.current_interior_state(simulator)
eq = simulator.system.equation 
Bvals = [B_cell(eq, i) for i in eachindex(x)]

# Conserved vars: (h1, q1, w, q2)
h1_0 = collect(initial_state.h1)
q1_0 = collect(initial_state.q1)    
w_0  = collect(initial_state.w)     
q2_0 = collect(initial_state.q2)    

h2_0 = w_0 .- Bvals

# Plot initial layers and bottom
lines!(ax_h, x, h1_0, label=L"h_{1,0}(x)")
lines!(ax_h, x, h2_0, label=L"h_{2,0}(x)")
lines!(ax_h, x, Bvals, linestyle=:dash, label=L"B(x)")

# Plot initial momenta
lines!(ax_q, x, q1_0, label=L"q_{1,0}(x)")
lines!(ax_q, x, q2_0, label=L"q_{2,0}(x)")

axislegend(ax_h, position=:lt)
axislegend(ax_q, position=:lt)
f


# --- Run simulation ---
@time SinFVM.simulate_to_time(simulator, T)

# --- Final state ---
result = SinFVM.current_interior_state(simulator)

h1 = collect(result.h1)
q1 = collect(result.q1)   
w  = collect(result.w)    
q2 = collect(result.q2)  

h2 = w .- Bvals

# Plot final results on top
lines!(ax_h, x, h1, linestyle=:dot, linewidth=6, label=L"h_1^{\Delta x}(x,t)")
lines!(ax_h, x, h2, linestyle=:dot, linewidth=6, label=L"h_2^{\Delta x}(x,t)")

lines!(ax_q, x, q1, linestyle=:dashdot, linewidth=6, label=L"q_1^{\Delta x}(x,t)")
lines!(ax_q, x, q2, linestyle=:dashdot, linewidth=6, label=L"q_2^{\Delta x}(x,t)")

axislegend(ax_h, position=:lt)
axislegend(ax_q, position=:lt)
f
