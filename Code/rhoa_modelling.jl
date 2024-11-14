### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f7844b00-8a69-493d-85f6-e98d534e9c32
begin
	import Pkg
	Pkg.activate(".")
	using Plots, PlutoUI, OrdinaryDiffEq, ModelingToolkit
end

# ╔═╡ 11f664e4-c2fb-407e-b97e-927ed7894b9b
md"""
### Environment setup:
"""

# ╔═╡ 93e6b659-907a-448d-a789-d9af70d037d4
md"""
# RhoA activation/deactivation model
This notebook contains various ODE systems intended to model experimental RhoA dynamics described in the paper "GTPase activating protein DLC1 spatio-temporally regulates Rho signaling" (Max Heydasch et al.).
"""

# ╔═╡ 80a186ba-3a47-4185-9ed4-62dc87369e05
md"""
### Global Assumptions:
- Total protein amount is conserved
- Biosensor detection rate negligible
- Michaelis-Menten dynamics for activation/deactivation mediated by protein interaction
"""

# ╔═╡ 63f0455f-c3fb-4a69-ac98-814f09c931ed
md"""
## Simple linear Model
### Specific Assumptions:
- Light signal translates into faster activation rate for short period of time (pulse)
- RhoA dynamics explained by simple linear ODE (activation/deactivation rate) 

"""

# ╔═╡ 922a83f7-6034-4fa3-8946-31bbb972d2b8
# Define light signal as symbolic function and variables for ODE

begin
	@independent_variables t
	D = Differential(t)
	
	# Light signal function
	f(t) = 5 * (0 <= t <= 1) + (1 < t) + (t < 0)
	@register_symbolic f(t)

	# Use hyperbolic tangent to have continuous derivatives for data fitting as light signal function
	f_tanh(t) = 1 + 5 * (tanh(100*(t))/2 - tanh(100*(t-1))/2) 
	@register_symbolic f_tanh(t)
	
end

# ╔═╡ 392b5c60-dd2f-42dd-8459-48db2b9bd3c4
# Define linear model using ModelingToolkit model
@mtkmodel RhoA_linear_dynamics begin

	@parameters begin
		#Kact_Rho
		Kinact_Rho
		steady
	end
	
	@variables begin
		S(t)
		aRhoA(t) 
	end
	
	@equations begin
		S ~ f_tanh(t)
		D(aRhoA) ~ steady * Kinact_Rho*S - Kinact_Rho * aRhoA 
	end
end

# ╔═╡ dc12a038-69ff-4072-b3af-265c485a7ec6
@mtkbuild RhoA_linear_model = RhoA_linear_dynamics()

# ╔═╡ 54356668-89e9-43b8-b675-cecde1b314ff
md"""
## GEF activation Model (non-linear)
### Specific Assumptions
- Light signal translates into higher aGEF initial conditions (pulse)
- Model incorporates RhoA activation by GEF
"""

# ╔═╡ 3e85a083-d7d0-45e3-bca0-1b733d635875
@mtkmodel RhoA_GEF_dynamics begin

	@parameters begin
		Kinact_GEF
		#Kact_GEF
		steady
		
		Kcat_aGEF
		Km_aGEF

		Kinact_RhoA
	end
	
	@variables begin
		aGEF(t)
		aRhoA(t)
		S(t)
	end
	
	@equations begin
		D(aGEF) ~ steady * Kinact_GEF * S - Kinact_GEF * aGEF
		D(aRhoA) ~ Kcat_aGEF * aGEF * (1 - aRhoA)/(Km_aGEF + (1 - aRhoA)) - Kinact_RhoA * aRhoA
		S ~ f_tanh(t)
	end
end

# ╔═╡ 6a550f32-f075-4f3e-9532-9579a149cd79
@mtkbuild RhoA_GEF_model = RhoA_GEF_dynamics()

# ╔═╡ 570eb272-ca98-4d0b-be36-5bda7483c529
md"""
## Simple GEF-GAP non-linear Model
### Specific Assumptions
- Light signal translates into higher aGEF initial conditions (pulse)
- Model incorporates RhoA activation by GEF and RhoA inhibition by GAP
"""

# ╔═╡ 62b20de5-0965-47b7-a20a-8d29761f2b4f
@mtkmodel RhoA_GEF_GAP_dynamics begin
	
	@parameters begin
		Kinact_GEF
		steady
		#Kact_GEF

		Kinact_GAP
		Kact_GAP
		
		Kcat_aGEF
		Km_aGEF

		Kcat_aGAP
		Km_aGAP

	end
	
	@variables begin
		aGEF(t)
		aGAP(t)
		aRhoA(t)
		S(t)
	end
	
	@equations begin
		D(aGEF) ~ steady * Kinact_GEF * S - Kinact_GEF * aGEF
		D(aGAP) ~ Kact_GAP - Kinact_GAP * aGAP
		D(aRhoA) ~ Kcat_aGEF * aGEF * (1 - aRhoA)/(Km_aGEF + (1 - aRhoA)) - Kcat_aGAP * aGAP * (aRhoA)/(Km_aGAP + (aRhoA))
		S ~ f_tanh(t)
	end
end

# ╔═╡ 57e765eb-9710-408c-b7f9-b64bebbbda8b
@mtkbuild RhoA_GEF_GAP_model = RhoA_GEF_GAP_dynamics()

# ╔═╡ e793fd40-2865-4328-ac03-87c64eead843
md"""
## Simple GAP negative feedback Model
### Specific Assumptions
- Light signal translates into higher aGEF initial conditions (pulse)
- Model incorporates RhoA activation by GEF and RhoA inhibition by GAP
- Negative autoregulation of activated RhoA via GAP activation
"""

# ╔═╡ 431091f3-4c0f-4c93-85a5-ef5aac765f97
@mtkmodel RhoA_GAP_FL_dynamics begin
	
	@parameters begin
		Kinact_GEF
		Kact_GEF

		Kinact_GAP
		Kcat_FL
		
		Kcat_aGEF
		Km_aGEF

		Kcat_aGAP
		Km_aGAP

		Km_FL
		n
	end
	
	@variables begin
		aGEF(t)
		aGAP(t)
		aRhoA(t)
		S(t)
	end
	
	@equations begin
		D(aGEF) ~ Kact_GEF * S - Kinact_GEF * aGEF
		D(aGAP) ~ Kcat_FL*((10 - aGAP) * aRhoA)/(Km_FL + (10 - aGAP)) - Kinact_GAP * aGAP
		D(aRhoA) ~ Kcat_aGEF * aGEF * (10 - aRhoA)/(Km_aGEF + (10 - aRhoA)) - (Kcat_aGAP * aGAP * aRhoA/(Km_aGAP + aRhoA))
		S ~ f_tanh(t)
	end
end

# ╔═╡ 3dda9a99-ee72-4714-b4ef-1d7c85cdddca
@mtkbuild RhoA_GAP_FL_model = RhoA_GAP_FL_dynamics()

# ╔═╡ 409bc225-581e-4a7a-93d5-26c57476da55
md"""
#### With Negative FL for DLC1 Factor
"""

# ╔═╡ 8c60a065-70ef-4858-9c87-6a004a1d6fad
parameters(RhoA_GAP_FL_model)

# ╔═╡ 03365ac1-15e3-4a35-986b-61483caf74b8
md"""
#### With Negative FL for X Factor
"""

# ╔═╡ 4e859e56-df11-4c0d-9557-2c879cfaab97
md"""
## Combined model with X and DLC1 negative feedback
"""

# ╔═╡ 29906a19-f5a7-4634-b2dc-0721b41eb40b
@mtkmodel RhoA_final_dynamics begin
	@parameters begin
		Kinact_GEF
		Kact_GEF

		Kinact_X
		Kcat_aX
		Km_aX
		Km_FL_X
		Kcat_FL_X
		
		Kcat_aGEF
		Km_aGEF

		Kact_DLC1
		Kinact_DLC1
		Kcat_aDLC1
		Km_aDLC1
		Km_FL_DLC1
		Kcat_FL_DLC1
	end
	
	@variables begin
		aRhoA(t) 
		aGEF(t) 
		aDLC1(t) 
		aX(t)
		S(t)
	end
	
	@equations begin
		D(aGEF) ~ Kact_GEF*S - Kinact_GEF * aGEF
		
		D(aX) ~ Kcat_FL_X*((1-aX) * aRhoA)/(Km_FL_X + (1-aX)) - Kinact_X * aX
		
		D(aDLC1) ~ Kcat_FL_DLC1*((1-aDLC1) * aRhoA)/(Km_FL_DLC1 + (1-aDLC1)) + Kact_DLC1 - Kinact_DLC1 * aDLC1
				
		D(aRhoA) ~ Kcat_aGEF * aGEF * (1 - aRhoA)/(Km_aGEF + (1 - aRhoA)) - (Kcat_aDLC1 * aDLC1 * aRhoA/(Km_aDLC1 + aRhoA)) - (Kcat_aX * aX * aRhoA/(Km_aX + aRhoA))
		
		S ~ f(t)
	end
end

# ╔═╡ 3dd2e52a-e077-44c7-b3d2-6be6f738a276
@mtkbuild RhoA_final_model = RhoA_final_dynamics()

# ╔═╡ 1704361e-3987-4465-9cc3-7418d4a34091
md"""
##### Conclusion:
- Combined model does not explain the dynamics assuming DLC1 and X factor (GAP) both present when DLC1 not KO. Factor X always speeds up deactivation rate.
- Two possible explanations:
    - Possible that a delay factor is needed for X factor to come (through effector). Model?
    - Possible that DLC1 KO cells have adapted to recruit other GAP permanently in the cell permanently with different dynamics.
	
"""

# ╔═╡ aee667c6-d886-4355-8dec-75abb2c63deb
function parameter_input(parameters::Vector)
	
	return PlutoUI.combine() do Child
		
		inputs = [
			md""" $(param[1]): $(
				Child(param[1], Slider(0:param[2]:param[3], default=param[4]))
			)"""
			
			for param in parameters
		]
		
		md"""
		#### Parameters
		$(inputs)
		"""
	end
end

# ╔═╡ 4109a142-b1ad-4efc-a0ca-784e16ab44b1
# Define the parameters as sliders over given range and with default value
@bind linear_parameter parameter_input([
	("aRhoa_0", 0.001, 2, 0.1), 
	("Kinact", 0.01, 0.5, 0.1)])

#("Kact", 0.001, 0.1, 0.01),

# ╔═╡ 24a05210-10c8-4c52-818f-eae902c52162
# Solve ODE problem with given initial conditions, time interval and parameters 
begin
	# Define problem ([[initial conditions], (time interval), [parameters]])
	linear_prob = ODEProblem(RhoA_linear_model, 
		[RhoA_linear_model.aRhoA => linear_parameter.aRhoa_0], (0.0, 200.0), [RhoA_linear_model.steady => linear_parameter.aRhoa_0, 
			RhoA_linear_model.Kinact_Rho => linear_parameter.Kinact])
	# RhoA_linear_model.Kact_Rho => linear_parameter.Kact

	# Solve problem with default solver (can change solver method)
	linear_sol = solve(linear_prob, saveat=0.05, dtmax = 1)
end

# ╔═╡ 2169896e-55d0-4622-93da-6a70a41785f7
plot(linear_sol)

# ╔═╡ 5a2d87c5-f08c-463d-8f78-b0773007381a
# Define the parameters as sliders over given range and with default value
@bind GEF_parameter parameter_input([
	("aGEF_0", 0.001, 1, 0.1),
	("Kinact_GEF", 0.001, 0.1, 0.041), 
	("Kcat_aGEF", 0.001, 0.5, 0.119),
	("Km_aGEF", 0.1, 1, 0.2),
	("Kinact_RhoA", 0.001, 0.5, 0.098)])


# Previous default: 
# 	("Kinact_GEF", 0.001, 0.1, 0.01), 
#   ("Kact_GEF", 0.001, 0.01, 0.001),
#	("Kcat_aGEF", 0.001, 0.5, 0.136),
#	("Km_aGEF", 0.1, 1, 0.5),
#	("Kinact_RhoA", 0.001, 0.5, 0.089)])

# ╔═╡ 881254b5-a86d-4438-943a-f7c7686be2d6
begin
	GEF_prob = ODEProblem(RhoA_GEF_model, 
		[RhoA_GEF_model.aGEF => GEF_parameter.aGEF_0, RhoA_GEF_model.aRhoA => 0.1], 
		(0.0, 200.0), 
		[RhoA_GEF_model.steady => GEF_parameter.aGEF_0, 
			RhoA_GEF_model.Kinact_GEF => GEF_parameter.Kinact_GEF, 
			RhoA_GEF_model.Kcat_aGEF => GEF_parameter.Kcat_aGEF, 
			RhoA_GEF_model.Km_aGEF => GEF_parameter.Km_aGEF,
			RhoA_GEF_model.Kinact_RhoA => GEF_parameter.Kinact_RhoA])
	
	GEF_sol = solve(GEF_prob, saveat=1, dtmax = 0.5)
	plot(GEF_sol, idxs=2)
end

# ╔═╡ bafe6303-ac61-4dc7-b77a-c8086d0ac54e
@bind GEF_GAP_parameters parameter_input([
	("aGEF_0", 0.001, 1, 0.093),
	("Kinact_GEF", 0.001, 0.1, 0.082),
	("Kact_GAP", 0.001, 0.01, 0.002),
	("Kinact_GAP", 0.001, 0.1, 0.02),
	("Kcat_aGEF", 0.001,0.1, 0.054),
	("Km_aGEF", 0.1, 1, 0.3),
	("Kcat_aGAP", 0.001, 0.1, 0.076),
	("Km_aGAP", 0.1, 1, 0.1)
])

# 	("Kact_GEF", 0.001, 0.01, 0.002)
#	("Kinact_GEF", 0.001, 0.1, 0.02),
#	("Kinact_GAP", 0.001, 0.1, 0.019),
#	("Kact_GAP", 0.001, 0.01, 0.001),
#	("Kcat_aGEF", 0.001,0.1, 0.034),
#	("Km_aGEF", 0.1, 1, 0.8),
#	("Km_aGAP", 0.1, 1, 0.1)
#	("Kcat_aGAP", 0.001, 0.1, 0.072),

# ╔═╡ 3ff9213a-27d2-475b-8d2f-8314be35a88d
begin
	GEF_GAP_prob = ODEProblem(RhoA_GEF_GAP_model, 
		[RhoA_GEF_GAP_model.aGEF => GEF_GAP_parameters.aGEF_0,     # 0.2
			RhoA_GEF_GAP_model.aRhoA => 0.1,
			RhoA_GEF_GAP_model.aGAP => 0.1], 
		(0.0, 350.0), 
		[RhoA_GEF_GAP_model.steady => GEF_GAP_parameters.aGEF_0, 
			RhoA_GEF_GAP_model.Kinact_GEF => GEF_GAP_parameters.Kinact_GEF, 
			RhoA_GEF_GAP_model.Kcat_aGEF => GEF_GAP_parameters.Kcat_aGEF, 
			RhoA_GEF_GAP_model.Km_aGEF => GEF_GAP_parameters.Km_aGEF, 
			RhoA_GEF_GAP_model.Kact_GAP => GEF_GAP_parameters.Kact_GAP,
			RhoA_GEF_GAP_model.Kinact_GAP => GEF_GAP_parameters.Kinact_GAP,
			RhoA_GEF_GAP_model.Kcat_aGAP => GEF_GAP_parameters.Kcat_aGAP, 
			RhoA_GEF_GAP_model.Km_aGAP => GEF_GAP_parameters.Km_aGAP
		])
	
	GEF_GAP_sol = solve(GEF_GAP_prob)
	plot(GEF_GAP_sol, idxs=3)
	#plot(GEF_GAP_sol)
end

# ╔═╡ 2887940d-d263-4366-b0b5-c1b954aafbb8
@bind GAP_FL_parameters parameter_input([
	("Kact_GEF", 0.001, 0.01, 0.002),
	("Kinact_GEF", 0.001, 0.1, 0.02),
	("Kcat_FL", 0.001, 0.1, 0.03),
	("Km_FL", 0.001, 0.1, 0.1), 
	("Kinact_GAP", 0.001, 0.1, 0.009),
	("Kcat_aGEF", 0.001, 0.1, 0.026),
	("Km_aGEF", 0.1, 1, 0.4),
	("Kcat_aGAP", 0.001, 0.5, 0.047),
	("Km_aGAP", 0.1, 1, 0.8),
	("n", 1, 10, 1)
])

# ╔═╡ 484340a8-3b5b-4a86-b4ee-12d9167559ff
begin
	GAP_FL_prob = ODEProblem(RhoA_GAP_FL_model, 
		[RhoA_GAP_FL_model.aGEF => 0.1, 
			RhoA_GAP_FL_model.aRhoA => 0.1,
			RhoA_GAP_FL_model.aGAP => 0.2], 
		(0.0, 250.0), 
		[RhoA_GAP_FL_model.Kinact_GEF => GAP_FL_parameters.Kinact_GEF,
			RhoA_GAP_FL_model.Kact_GEF => GAP_FL_parameters.Kact_GEF,  
			RhoA_GAP_FL_model.Kcat_aGEF => GAP_FL_parameters.Kcat_aGEF, 
			RhoA_GAP_FL_model.Km_aGEF => GAP_FL_parameters.Km_aGEF, 
			RhoA_GAP_FL_model.Kcat_FL => GAP_FL_parameters.Kcat_FL,
			RhoA_GAP_FL_model.Kinact_GAP => GAP_FL_parameters.Kinact_GAP,
			RhoA_GAP_FL_model.Kcat_aGAP => GAP_FL_parameters.Kcat_aGAP, 
			RhoA_GAP_FL_model.Km_aGAP => GAP_FL_parameters.Km_aGAP,
			RhoA_GAP_FL_model.Km_FL => GAP_FL_parameters.Km_FL,
			RhoA_GAP_FL_model.n => GAP_FL_parameters.n
		])
	
	GAP_FL_sol = solve(GAP_FL_prob)
	plot(GAP_FL_sol, idxs=3)
	#plot(GAP_FL_sol)
end

# ╔═╡ 9ca3b905-ecef-4e4b-b892-290c2008f28e
@bind X_FL_parameters parameter_input([
	("Kact_GEF", 0.001, 0.01, 0.002),
	("Kinact_GEF", 0.001, 0.1, 0.02),
	("Kcat_FL", 0.001, 0.1, 0.007),
	("Km_FL", 0.001, 0.1, 0.093), 
	("Kinact_GAP", 0.001, 0.1, 0.021),
	("Kcat_aGEF", 0.001, 0.1, 0.026),
	("Km_aGEF", 0.1, 1, 0.4),
	("Kcat_aGAP", 0.001, 0.5, 0.495),
	("Km_aGAP", 0.1, 1, 0.8),
	("n", 1, 10, 1)
])


# ╔═╡ 9543437a-e965-4d99-a952-c15b0af41850
begin
	X_FL_prob = ODEProblem(RhoA_GAP_FL_model, 
		[RhoA_GAP_FL_model.aGEF => 0.1, 
			RhoA_GAP_FL_model.aRhoA => 0.1,
			RhoA_GAP_FL_model.aGAP => 0], 
		(0.0, 200.0), 
		[RhoA_GAP_FL_model.Kact_GEF => X_FL_parameters.Kact_GEF, 
			RhoA_GAP_FL_model.Kinact_GEF => X_FL_parameters.Kinact_GEF, 
			RhoA_GAP_FL_model.Kcat_aGEF => X_FL_parameters.Kcat_aGEF, 
			RhoA_GAP_FL_model.Km_aGEF => X_FL_parameters.Km_aGEF, 
			RhoA_GAP_FL_model.Kcat_FL => X_FL_parameters.Kcat_FL,
			RhoA_GAP_FL_model.Kinact_GAP => X_FL_parameters.Kinact_GAP,
			RhoA_GAP_FL_model.Kcat_aGAP => X_FL_parameters.Kcat_aGAP, 
			RhoA_GAP_FL_model.Km_aGAP => X_FL_parameters.Km_aGAP,
			RhoA_GAP_FL_model.Km_FL => X_FL_parameters.Km_FL,
			RhoA_GAP_FL_model.n => X_FL_parameters.n
		])
	
	X_FL_sol = solve(X_FL_prob)
	plot(X_FL_sol, idxs=3)
	#plot(X_FL_sol)
end

# ╔═╡ 7f388cbe-8226-4384-8320-2284bb70bc6a
@bind final_param parameter_input([
	("Kinact_X", 0.001, 0.1, 0.021), 
	("Kcat_aX", 0.001, 0.5, 0.495),
	("Km_aX", 0.1, 1, 0.8),
	("Kcat_FL_X", 0.001, 0.1, 0.007),
	("Km_FL_X", 0.001, 0.1, 0.093),
	("Kact_DLC1", 0.0001, 0.01, 0.002),
	("Kinact_DLC1", 0.0001, 0.1, 0.0347),
	("Kcat_aDLC1", 0.001,0.1, 0.009),
	("Km_aDLC1", 0.1, 1, 0.2),
	("Kcat_FL_DLC1", 0.001, 0.1, 0.011),
	("Km_FL_DLC1", 0.001, 0.9, 0.1617),
])

# ╔═╡ 47de9624-16c5-4499-9b54-e3fff3a56e21
begin
	final_prob = ODEProblem(RhoA_final_model, 
		[RhoA_final_model.aGEF => 0.1, 
			RhoA_final_model.aRhoA => 0.1,
			RhoA_final_model.aDLC1 => 0.1,
			RhoA_final_model.aX => 0
		], 
		(0.0, 200.0), 
		[RhoA_final_model.Kact_GEF => 0.002, 
			RhoA_final_model.Kinact_GEF => 0.02, 
			RhoA_final_model.Kcat_aGEF => 0.026, 
			RhoA_final_model.Km_aGEF => 0.4,
			RhoA_final_model.Kinact_X => final_param.Kinact_X,
			RhoA_final_model.Kcat_aX => final_param.Kcat_aX,
			RhoA_final_model.Km_aX => final_param.Km_aX,
			RhoA_final_model.Kcat_FL_X => final_param.Kcat_FL_X,
			RhoA_final_model.Km_FL_X => final_param.Km_FL_X,
			RhoA_final_model.Kact_DLC1 => final_param.Kact_DLC1,
			RhoA_final_model.Kinact_DLC1 => final_param.Kinact_DLC1,
			RhoA_final_model.Kcat_aDLC1 => final_param.Kcat_aDLC1,
			RhoA_final_model.Km_aDLC1 => final_param.Km_aDLC1,
			RhoA_final_model.Kcat_FL_DLC1 => final_param.Kcat_FL_DLC1, 
			RhoA_final_model.Km_FL_DLC1 => final_param.Km_FL_DLC1
		])
	
	final_sol = solve(final_prob)
	plot(final_sol, idxs=4)
	#plot(final_sol)
end

# ╔═╡ Cell order:
# ╟─11f664e4-c2fb-407e-b97e-927ed7894b9b
# ╟─f7844b00-8a69-493d-85f6-e98d534e9c32
# ╟─93e6b659-907a-448d-a789-d9af70d037d4
# ╟─80a186ba-3a47-4185-9ed4-62dc87369e05
# ╟─63f0455f-c3fb-4a69-ac98-814f09c931ed
# ╟─922a83f7-6034-4fa3-8946-31bbb972d2b8
# ╟─392b5c60-dd2f-42dd-8459-48db2b9bd3c4
# ╠═dc12a038-69ff-4072-b3af-265c485a7ec6
# ╟─24a05210-10c8-4c52-818f-eae902c52162
# ╟─4109a142-b1ad-4efc-a0ca-784e16ab44b1
# ╠═2169896e-55d0-4622-93da-6a70a41785f7
# ╟─54356668-89e9-43b8-b675-cecde1b314ff
# ╟─3e85a083-d7d0-45e3-bca0-1b733d635875
# ╟─6a550f32-f075-4f3e-9532-9579a149cd79
# ╟─5a2d87c5-f08c-463d-8f78-b0773007381a
# ╟─881254b5-a86d-4438-943a-f7c7686be2d6
# ╟─570eb272-ca98-4d0b-be36-5bda7483c529
# ╟─62b20de5-0965-47b7-a20a-8d29761f2b4f
# ╠═57e765eb-9710-408c-b7f9-b64bebbbda8b
# ╟─bafe6303-ac61-4dc7-b77a-c8086d0ac54e
# ╟─3ff9213a-27d2-475b-8d2f-8314be35a88d
# ╟─e793fd40-2865-4328-ac03-87c64eead843
# ╟─431091f3-4c0f-4c93-85a5-ef5aac765f97
# ╠═3dda9a99-ee72-4714-b4ef-1d7c85cdddca
# ╟─409bc225-581e-4a7a-93d5-26c57476da55
# ╠═2887940d-d263-4366-b0b5-c1b954aafbb8
# ╟─484340a8-3b5b-4a86-b4ee-12d9167559ff
# ╠═8c60a065-70ef-4858-9c87-6a004a1d6fad
# ╟─03365ac1-15e3-4a35-986b-61483caf74b8
# ╟─9ca3b905-ecef-4e4b-b892-290c2008f28e
# ╟─9543437a-e965-4d99-a952-c15b0af41850
# ╟─4e859e56-df11-4c0d-9557-2c879cfaab97
# ╟─29906a19-f5a7-4634-b2dc-0721b41eb40b
# ╟─3dd2e52a-e077-44c7-b3d2-6be6f738a276
# ╟─7f388cbe-8226-4384-8320-2284bb70bc6a
# ╟─47de9624-16c5-4499-9b54-e3fff3a56e21
# ╟─1704361e-3987-4465-9cc3-7418d4a34091
# ╟─aee667c6-d886-4355-8dec-75abb2c63deb
