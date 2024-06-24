using StructuralIdentifiability, ModelingToolkit

# This script analyses the global identifiability of the parameters of the ODE system used to model RhoA dynamics

# Define ODE system using MTK
@variables t
D = Differential(t)
    
@mtkmodel RhoA_KO_dynamics begin

    @parameters begin
        Kinact_GEF
        Kcat_aGEF
        Km_aGEF
        Kcat_FL
        Km_FL
        Kinact_GAP
        Kcat_aGAP
        Km_aGAP
        GEF0
        GAP0
    end

    @variables begin
        aGEF(t), [bounds = (0, 10)]
        aGAP(t), [bounds = (0, 10)]
        aRhoA(t), [bounds = (0, 10)]
        y(t), [output = true]
    end
    
    @equations begin
        D(aGEF) ~ Kinact_GEF * (GEF0 - aGEF)
        D(aGAP) ~ Kcat_FL * ((aRhoA-1)^2*(tanh(100*(aRhoA-1))-tanh(100*(aRhoA-10)))/2 * (10 - aGAP))/(Km_FL + (10 - aGAP)) + Kinact_GAP * (GAP0-aGAP)
        #D(aGAP) ~ Kcat_FL * (aRhoA-1) + Kinact_GAP * (0.5-aGAP)
        #D(aGAP) ~ Kcat_FL * ((10 - aGAP) * (aRhoA-1)^6)/(Km_FL + (10 - aGAP)) + Kinact_GAP * (0.5-aGAP)
        D(aRhoA) ~ Kcat_aGEF * (aGEF-GEF0)*(tanh(100*(aGEF-GEF0))-tanh(100*(aGEF-10)))/2 * (10 - aRhoA)/(Km_aGEF + (10 - aRhoA)) - Kcat_aGAP * (aGAP-GAP0)*(tanh(100*(aGAP-GAP0))-tanh(100*(aGAP-10)))/2 * (aRhoA)/(Km_aGAP + (aRhoA))
        #D(aRhoA) ~ Kcat_aGEF * (aGEF-0.5) - Kcat_aGAP * (aGAP-0.5)
        #D(aRhoA) ~ Kcat_aGEF * (aGEF-0.5) * (10 - aRhoA)/(Km_aGEF + (10 - aRhoA)) - Kcat_aGAP * (aGAP-0.5) #* aRhoA/(Km_aGAP + aRhoA)
        y ~ aRhoA
    end
end

@mtkbuild RhoA_KO_model = RhoA_KO_dynamics()

# Assess global identifiability without initial conditions
global_identifiability = assess_identifiability(RhoA_KO_model, measured_quantities=[RhoA_KO_model.y ~ RhoA_KO_model.aRhoA], prob_threshold = 0.99)
println(global_identifiability)


# Define ODE system using ODEmodel to assess parameter identifiability with initial conditions
RhoA_ode = @ODEmodel(
    aGEF'(t) = Kinact_GEF * (0.5 - aGEF(t)),
    aGAP'(t) ~ (Kcat_FL * ((aRhoA(t)-1)^2*(tanh(100*(aRhoA(t)-1))-tanh(100*(aRhoA(t)-10)))/2 * 
            (10 - aGAP(t)))/(Km_FL + (10 - aGAP(t))) + Kinact_GAP * (0.5-aGAP(t)))

    aRhoA'(t) = Kcat_aGEF * (aGEF(t)-0.5)*(tanh(100*(aGEF(t)-0.5))-tanh(100*(aGEF(t)-10)))/2 * (10 - aRhoA(t))/(Km_aGEF + (10 - aRhoA(t))) - Kcat_aGAP * (aGAP(t)-0.5)*(tanh(100*(aGAP(t)-0.5))-tanh(100*(aGAP(t)-10)))/2 * (aRhoA(t))/(Km_aGAP + (aRhoA(t)))
    
    y(t) = aRhoA(t)

    #aGAP'(t) = Kcat_FL * ((10 - aGAP(t)) * (aRhoA(t)-1)^6)/(Km_FL + (10 - aGAP(t))) + Kinact_GAP * (0.5-aGAP(t)),
    #aGAP'(t) = Kcat_FL * (aRhoA(t)-1)^6 + Kinact_GAP * (0.5 - aGAP(t)),
    #aRhoA'(t) = Kcat_aGEF * (aGEF(t)-0.5) * (10 - aRhoA(t))/(Km_aGEF + (10 - aRhoA(t))) - (Kcat_aGAP * (aGAP(t)-0.5) * aRhoA(t)/(Km_aGAP + aRhoA(t))),
    #aRhoA'(t) = Kcat_aGEF * (aGEF(t)-0.5) - (Kcat_aGAP * (aGAP(t)-0.5)),
)

# Assess global identifiability with initial conditions
global_identifiability_ic = assess_identifiability(RhoA_ode, prob_threshold = 0.99, known_ic=[aGAP, aGEF, aRhoA])
println("Parameter identifiability with initial conditions for all variables:")
println(global_identifiability_ic)