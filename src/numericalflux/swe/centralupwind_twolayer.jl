

struct CentralUpwindTwoLayer{E<:TwoLayerShallowWaterEquations1D} <: NumericalFlux
    eq::E
end

