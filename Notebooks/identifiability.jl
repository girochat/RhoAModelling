using StructuralIdentifiability, ModelingToolkit

# This script analyses the global identifiability of the parameters of the ODE system used to model RhoA dynamics

# Define ODE system using MTK
@independent_variables t
D = Differential(t)
    
@mtkmodel RhoA_dynamics begin

    @parameters begin
        Kon_GEF
        Koff_GEF
        Kon_GAP
        Koff_GAP
        Kon_Rho
        Koff_Rho
    end

    @variables begin
        aGEF(t), [bounds = (0, 10)]
        aGAP(t), [bounds = (0, 10)]
        aRhoA(t), [bounds = (0, 10)]
        y(t), [output = true]
    end
    
    @equations begin
        D(aGEF) ~ Kon_GEF - Koff_GEF * aGEF
        D(aGAP) ~ Kon_GAP * (10 - aGAP) * aRhoA - Koff_GAP * aGAP
        D(aRhoA) ~ Kon_Rho * aGEF * (10 - aRhoA) - Koff_Rho * aGAP * aRhoA
        y ~ aRhoA
    end
end

@mtkbuild RhoA_model = RhoA_dynamics()

# Assess global identifiability without initial conditions
global_identifiability = assess_identifiability(RhoA_model, measured_quantities=[RhoA_model.y ~ RhoA_model.aRhoA], prob_threshold = 0.99)
println("Parameter identifiability of RhoA model :")
println(global_identifiability)


# Define ODE system using ODEmodel to assess parameter identifiability with initial conditions
RhoA_ode = @ODEmodel(
    aGEF'(t) = Kon_GEF - Koff_GEF * aGEF(t),
    aGAP'(t) = Kon_GAP * (10 - aGAP(t)) * aRhoA(t) - Koff_GAP * aGAP(t),
    aRhoA'(t) = Kon_Rho * aGEF(t) * (10 - aRhoA(t)) - Koff_Rho * aGAP(t) * aRhoA(t),
    y(t) = aRhoA(t)
)

# Assess global identifiability with initial conditions
global_identifiability_ic = assess_identifiability(RhoA_ode, prob_threshold = 0.99, known_ic=[aGAP, aGEF, aRhoA])
println("Parameter identifiability with initial conditions for all variables:")
println(global_identifiability_ic)