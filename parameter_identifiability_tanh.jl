using StructuralIdentifiability, ModelingToolkit

# This script analyses the global identifiability of the parameters of the ODE system used to model RhoA dynamics
#################################################################################################################

# Approximate tanh(x) as a polynomial series: tanh(x) ≈ x - x^3 / 3 + x^5 / 5 in the following model

# Define ODE system using MTK
@independent_variables t
D = Differential(t)
    
@mtkmodel Rho_dynamics begin

    @parameters begin
        Kon_GEF
        Koff_GEF
        Kon_GAP
        Koff_GAP
        Kon_Rho
        Koff_Rho
    end

    @variables begin
        GEF(t), [bounds = (0, 10)]
        GAP(t), [bounds = (0, 10)]
        Rho(t), [bounds = (0, 10)]
        y(t), [output = true]
    end
    
    @equations begin
        D(GEF) ~ Kon_GEF - Koff_GEF * GEF
        D(GAP) ~ Kon_GAP * (1 - GAP) * (Rho-1) * ((100*(Rho-1) - (100*(Rho-1))^3 /3 + (100*(Rho-1))^5/5) - (100*(Rho-2) + (100*(Rho-2))^3 /3 - (100*(Rho-2))^5/5))/2 - Koff_GAP * GAP
        D(Rho) ~ Kon_Rho * GEF * (2 - Rho) - Koff_Rho * GAP * Rho 
        y ~ Rho
    end
end

@mtkbuild Rho_model = Rho_dynamics()

# Assess global identifiability without initial conditions
global_identifiability = assess_identifiability(Rho_model, measured_quantities=[Rho_model.y ~ Rho_model.Rho], prob_threshold = 0.99)
println("Parameter identifiability of RhoA model :")
println(global_identifiability)


# Define ODE system using ODEmodel to assess parameter identifiability with initial conditions
Rho_ode = @ODEmodel(
    GEF'(t) = Kon_GEF - Koff_GEF * GEF(t),
    GAP'(t) = Kon_GAP * (1 - GAP(t)) * (Rho(t)-1) * ((100*(Rho(t)-1) - (100*(Rho(t)-1))^3 /3 + (100*(Rho(t)-2))^5/5) - (100*(Rho(t)-2) + (100*(Rho(t)-2))^3 /3 - (100*(Rho(t)-2))^5/5))/2 - Koff_GAP * GAP(t),
    Rho'(t) = Kon_Rho * GEF(t) * (2 - Rho(t)) - Koff_Rho * GAP(t) * Rho(t),
    y(t) = Rho(t)
)

# Assess global identifiability with initial conditions
global_identifiability_ic = assess_identifiability(Rho_ode, prob_threshold = 0.99, known_ic=[GAP, GEF, Rho])
println("Parameter identifiability with initial conditions for all variables:")
println(global_identifiability_ic)