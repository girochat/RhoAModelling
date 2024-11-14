### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 40440a7c-70a9-4c35-a234-cdb419a803d1
begin
	import Pkg
	Pkg.activate(".")
	using OrdinaryDiffEq, Optimization, OptimizationPolyalgorithms, SciMLSensitivity, ModelingToolkit, Optim, ForwardDiff, Plots
end

# ╔═╡ feb8a0ed-3cc6-4d94-87af-2f3d8ffffb06
md"""
### Environment setup
"""

# ╔═╡ a97176f9-7dd2-4870-9a80-b91e25ba8b97
md"""
## Data fitting for RhoA activation/deactivation model
Data source: Lucien Hinderling, PertzLab
"""

# ╔═╡ ea2c40f1-f338-4879-a502-69aa6a35e002
md"""
#### 1: Fit linear model to simulated data:
Data simulation of linear activation/deactivation model and parameter optimisation to fit the model to the simulated data.
"""

# ╔═╡ e44a2ff8-7326-4080-813e-ffbeb4c4b648
md"""
###### Light signal function
To use the gradient for parameter optimisation the ODE system needs to be composed of continuous and differentiable function to avoid instability with the ODE solver. Instead of a step function, the light signal is modelled using a hyperbolic tangent.
"""

# ╔═╡ 52bdbc46-53fc-42ca-8fec-e43522c30063
# Define symbolic functions and variables for ODE model
begin
	@independent_variables t
	D = Differential(t)
	
	# Light signal function (step function)
	f(t) = 2 * (0 <= t <= 1) + (1 < t) + (t < 0)
	@register_symbolic f(t) 

	# Use hyperbolic tangent to have continuous derivatives for data fitting as light signal function
	f_tanh(t) = 1 + (tanh(100*(t))/2 - tanh(100*(t-1))/2) 
	@register_symbolic f_tanh(t)
end

# ╔═╡ 247533da-69ba-483c-8025-004c32a1bf9c
# Plot the light signal functions to check if the continuous approximation matches
# the step function
begin
	x = LinRange(-1, 2, 400)
	y1 = f.(x)
	y2 = f_tanh.(x)
	plot(x, [y1, y2], labels =["Step function" "Tanh"], lw = 2, title="Comparison of light signal functions")
end

# ╔═╡ e11c0ef1-18a8-4a48-b9ae-2e28755cafad
# Define RhoA linear model using ModelingToolkit model
@mtkmodel RhoA_linear_dynamics begin

	@parameters begin
		Kinact_Rho
	end
	
	@variables begin
		S(t)
		aRhoA(t) 
	end
	
	@equations begin
		S ~ f_tanh(t)
		D(aRhoA) ~ Kinact_Rho *(0.5 * S - aRhoA)
	end
end

# ╔═╡ 06601163-39a2-4daa-a39b-900d746bc9df
# Build model
@mtkbuild RhoA_linear_model = RhoA_linear_dynamics()

# ╔═╡ 4da074c7-6116-468d-a56d-4e8935b1bddb
# Simulate data for true parameters (kact = 0.009, kinact = 0.09)
begin

	# Note : 
	# ModelingToolkit allow to generate symbolic Jacobian from built model. It can then be used to solve ODE problems faster with jac = true option.
	ModelingToolkit.generate_jacobian(RhoA_linear_model)[2]
	
	data_simul_prob = ODEProblem(RhoA_linear_model, 
		[RhoA_linear_model.aRhoA => 0.5], (0.0, 200.0), [0.24], jac = true)
	
	data_simul_sol = solve(data_simul_prob, saveat=1, dtmax=0.5)
	
	# Import the data as array
	simul_data = data_simul_sol[RhoA_linear_model.aRhoA]
end

# ╔═╡ dd6e79d3-a652-4179-a3c6-dcd57916ceba
plot(data_simul_sol)

# ╔═╡ bd48010a-96f9-4922-b86e-bacb1d474568
# Define MSE loss function to estimate parameters
function MSE_loss(new_parameters)
	
	# Update the ODE problem with the new set of parameters
	new_ODE_prob = remake(data_simul_prob, p=new_parameters)

	# Solve the new ODE problem (saveat = numerical estimate time interval)
	new_sol = solve(new_ODE_prob, saveat = 1, dtmax=0.5)
	new_sol_array = new_sol[RhoA_linear_model.aRhoA]

	# Determine the loss by computing the sum of squares
	if length(simul_data) == length(new_sol_array)
		loss = sum(abs2, new_sol_array .- simul_data)
	else
		loss = 1000
	end

	return loss
end

# ╔═╡ beacbe18-63b7-4f41-b69b-aca8651a6937
# Define the optimisation problem
begin
	# Define optimization function and differentiation method
	optimisation_f = Optimization.OptimizationFunction((x, p) -> MSE_loss(x), 
		Optimization.AutoForwardDiff())

	# Initial parameter guess
	parameter_guess = [0.005, 0.01]

	# Keep track of loss values during iterations
	track_loss = []
	
	optimisation_prob = Optimization.OptimizationProblem(optimisation_f, parameter_guess)
end

# ╔═╡ fe29f3cc-2f90-4c33-b70f-d8ba91e950ea
# Define callback function that is run at each iteration
callback! = function(parameters, loss)

	# Add current loss value and parameter to tracking arrays
	push!(track_loss, loss)
	
	# Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
	return false
end

# ╔═╡ aac86f58-6c83-4810-80c4-e3e2c4ca137f
begin 
	# Clear the tracking array if not empty
	if length(track_loss) > 0
		deleteat!(track_loss, 1:length(track_loss))
	end

	# Run the optimisation
	optim_results = Optimization.solve(optimisation_prob, PolyOpt(), callback = callback!, maxiters = 100)
end

# ╔═╡ 0478f333-cb6e-499e-b81e-c1591da4fef8
# Plot the loss value at each iteration
plot(track_loss, title = "Evolution of the loss value", label = false)

# ╔═╡ da1c41c2-e494-417a-afe0-178334c36302
begin
	predicted_parameters = optim_results.u
	pred_sol = solve(remake(data_simul_prob, p=predicted_parameters), saveat=0.5, dtmax = 0.5)
	scatter(data_simul_sol, label = "Data simulation", lw=2, title = "Prediction (vs) simulation")
	plot!(pred_sol, label = "Prediction", lw=2)
end


# ╔═╡ 39b68055-0676-4254-9dbc-414580137130
md"""
#### 2: Fit RhoA negative feedback model to simulated data:
Data simulation of RhoA negative feedback activation/deactivation model and parameter optimisation to fit the model to the simulated data.
"""

# ╔═╡ d352c2f2-1dee-429e-ba4e-1e8860c1f987
# Define RhoA negative feedback on GAP model
@mtkmodel RhoA_GAP_FL_dynamics begin
	
	@parameters begin
		Kinact_GEF
		Kact_GEF

		Kcat_aGEF
		Km_aGEF
		Kinact_GAP
		
		Kcat_FL
		Km_FL

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
		D(aGEF) ~ Kact_GEF * S - Kinact_GEF * aGEF
		D(aGAP) ~ Kcat_FL*((1 - aGAP) * aRhoA)/(Km_FL + (1 - aGAP)) - Kinact_GAP * aGAP
		D(aRhoA) ~ Kcat_aGEF * aGEF * (1 - aRhoA)/(Km_aGEF + (1 - aRhoA)) - (Kcat_aGAP * aGAP * aRhoA/(Km_aGAP + aRhoA))
		S ~ f_tanh(t+150)
	end
end

# ╔═╡ 25d47ee9-64c3-410a-9b0b-6df36f106deb
@mtkbuild RhoA_GAP_FL_model = RhoA_GAP_FL_dynamics()

# ╔═╡ 05840396-3ee2-46d1-99aa-dda4c209a39f
# Simulate data for true parameters
begin
	FL_data_simul_prob = ODEProblem(RhoA_GAP_FL_model, 
		[RhoA_GAP_FL_model.aGEF => 0.1, 
			RhoA_GAP_FL_model.aRhoA => 0.1,
			RhoA_GAP_FL_model.aGAP => 0.2], 
		(0, 200.0), 
		[RhoA_GAP_FL_model.Kinact_GEF => 0.02, 
			RhoA_GAP_FL_model.Kact_GEF => 0.002, 
			RhoA_GAP_FL_model.Kinact_GAP => 0.009, 
			RhoA_GAP_FL_model.Kcat_FL => 0.03, 
			RhoA_GAP_FL_model.Kcat_aGEF => 0.026, 
			RhoA_GAP_FL_model.Km_aGEF => 0.4, 
			RhoA_GAP_FL_model.Kcat_aGAP => 0.047, 
			RhoA_GAP_FL_model.Km_aGAP => 0.8, 
			RhoA_GAP_FL_model.Km_FL => 0.1], jac = true)
	
	FL_data_simul_sol = solve(FL_data_simul_prob, saveat=1, dtmax=0.5)
	
	# Import the data as array
	FL_simul_data = FL_data_simul_sol[RhoA_GAP_FL_model.aRhoA]
end

# ╔═╡ 892e9d1e-296a-4493-83a0-5aa336906cc7
plot(FL_data_simul_sol, idxs=3)

# ╔═╡ 553f692b-d5ea-4700-8d26-ade61f71e3fd
# Define MSE loss function to estimate parameters
function FL_MSE_loss(new_parameters)
	
	# Update the ODE problem with the new set of parameters
	FL_new_ODE_prob = remake(FL_data_simul_prob, p=new_parameters)

	# Solve the new ODE problem (saveat = numerical estimate time interval)
	FL_new_sol = solve(FL_new_ODE_prob, Rosenbrock23(), saveat = 1, maxiters = 1e7, verbose = false, dtmax = 0.5)
	FL_new_sol_array = FL_new_sol[RhoA_GAP_FL_model.aRhoA]

	# Determine the loss by computing the sum of squares¨
	if length(FL_new_sol_array) == length(FL_simul_data)
		FL_loss = sum(abs2, FL_new_sol_array .- FL_simul_data)
	else
		FL_loss = 1000
	end

	return FL_loss
end

# ╔═╡ a3b95970-76fa-4e92-b479-d00e9e5f5d7e
# Define the optimisation problem
begin
	# Define optimization function and differentiation method
	FL_optimisation_f = Optimization.OptimizationFunction((x, p) -> FL_MSE_loss(x), 
		Optimization.AutoForwardDiff())

	# Initial parameter guess
	FL_parameter_guess = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

	# Keep track of parameters and loss values during iterations
	FL_track_loss = []

	FL_optimisation_prob = Optimization.OptimizationProblem(FL_optimisation_f, FL_parameter_guess)
end

# ╔═╡ 013d6c31-3e9c-4544-93a7-497e408b980a
# Define callback function that is run at each iteration
FL_callback! = function(parameters, loss)

	# Add current loss value and parameter to tracking arrays
	push!(FL_track_loss, loss)
	
	# Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
	return false
end

# ╔═╡ 9fc562d2-786d-4646-ad84-14429f49aa2c
begin 
	# Clear the tracking arrays if not empty
	if length(FL_track_loss) > 0
		deleteat!(FL_track_loss, 1:length(FL_track_loss))
	end

	# Run the optimisation
	FL_optim_results = Optimization.solve(FL_optimisation_prob, PolyOpt(), callback = FL_callback!, maxiters = 200)
end

# ╔═╡ 7356a98b-b003-4439-befd-69f7b7b62461
# Visualise optimisation statistics
FL_optim_results.original

# ╔═╡ 5fd77698-61d8-4b1f-a510-3c20fd4bcad4
# Plot the loss value at each iteration
plot(FL_track_loss, title = "Evolution of the loss value", label = false)

# ╔═╡ 34c4b535-fb45-4a75-875c-14cd333e38cf
# Plot predicted model compared to simulated data
begin
	FL_predicted_parameters = FL_optim_results.u
	FL_pred_sol = solve(remake(FL_data_simul_prob, p=FL_predicted_parameters), saveat=0.05, dtmax = 0.5)
	scatter(FL_data_simul_sol, label = "Data simulation", lw=2, title = "Prediction (vs) simulation", idxs=3)
	plot!(FL_pred_sol, label = "Prediction", lw=2, idxs=3)
end

# ╔═╡ Cell order:
# ╠═feb8a0ed-3cc6-4d94-87af-2f3d8ffffb06
# ╠═40440a7c-70a9-4c35-a234-cdb419a803d1
# ╟─a97176f9-7dd2-4870-9a80-b91e25ba8b97
# ╟─ea2c40f1-f338-4879-a502-69aa6a35e002
# ╟─e44a2ff8-7326-4080-813e-ffbeb4c4b648
# ╠═52bdbc46-53fc-42ca-8fec-e43522c30063
# ╠═247533da-69ba-483c-8025-004c32a1bf9c
# ╠═e11c0ef1-18a8-4a48-b9ae-2e28755cafad
# ╠═06601163-39a2-4daa-a39b-900d746bc9df
# ╠═4da074c7-6116-468d-a56d-4e8935b1bddb
# ╟─dd6e79d3-a652-4179-a3c6-dcd57916ceba
# ╠═bd48010a-96f9-4922-b86e-bacb1d474568
# ╠═beacbe18-63b7-4f41-b69b-aca8651a6937
# ╠═fe29f3cc-2f90-4c33-b70f-d8ba91e950ea
# ╠═aac86f58-6c83-4810-80c4-e3e2c4ca137f
# ╠═0478f333-cb6e-499e-b81e-c1591da4fef8
# ╟─da1c41c2-e494-417a-afe0-178334c36302
# ╟─39b68055-0676-4254-9dbc-414580137130
# ╟─d352c2f2-1dee-429e-ba4e-1e8860c1f987
# ╟─25d47ee9-64c3-410a-9b0b-6df36f106deb
# ╠═05840396-3ee2-46d1-99aa-dda4c209a39f
# ╟─892e9d1e-296a-4493-83a0-5aa336906cc7
# ╠═553f692b-d5ea-4700-8d26-ade61f71e3fd
# ╠═013d6c31-3e9c-4544-93a7-497e408b980a
# ╟─a3b95970-76fa-4e92-b479-d00e9e5f5d7e
# ╠═9fc562d2-786d-4646-ad84-14429f49aa2c
# ╠═7356a98b-b003-4439-befd-69f7b7b62461
# ╠═5fd77698-61d8-4b1f-a510-3c20fd4bcad4
# ╠═34c4b535-fb45-4a75-875c-14cd333e38cf
