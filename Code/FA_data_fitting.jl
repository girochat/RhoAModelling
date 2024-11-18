### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ cfa5e8b3-cfea-42bc-ab54-023d9de67850
begin
	# Environment setup
	import Pkg
	Pkg.activate(".")
	using OrdinaryDiffEq, Optimization, OptimizationPolyalgorithms, SciMLSensitivity, ModelingToolkit, Optim, ForwardDiff, Plots, DelimitedFiles
end

# ╔═╡ de0e5ef4-c53d-4659-9bc0-73f7b10e923d
md"""
### Environment setup
"""

# ╔═╡ a97176f9-7dd2-4870-9a80-b91e25ba8b97
md"""
## Data fitting of RhoA activation/deactivation dynamics at Focal Adhesions
This notebook contains the code used to fit an ODE model to the experimental data of RhoA activity obtained after local light stimulation at focal adhesions as described in the paper by Heydasch et al. \


Max Heydasch, Lucien Hinderling, Jakobus van Unen, Maciej Dobrzynski, Olivier Pertz
(2023) \
**GTPase activating protein DLC1 spatio-temporally regulates Rho signaling** \
eLife 12:RP90305 \
https://doi.org/10.7554/eLife.90305.1  

Data source: Lucien Hinderling, Pertz Lab, Institute of Cell Biology, University of Bern, Switzerland.
"""

# ╔═╡ 44e0f32e-3a41-44d6-bf44-0dc7d323c0b6
md"""
### Initial steps
##### - Light signal function
To use the gradient for parameter optimisation, the ODE system needs to be composed of continuous and differentiable function to avoid instability with the ODE solver. Instead of a step function, the light signal is modelled using a hyperbolic tangent.
"""

# ╔═╡ 52bdbc46-53fc-42ca-8fec-e43522c30063
# Define symbolic functions and variables for ODE model
begin
	@independent_variables t
	D = Differential(t)
	
	# Light signal function (step function)
	f(t) = (160 <= t <= 160.8)
	@register_symbolic f(t) 

	# Use hyperbolic tangent as continuous approximation of light signal function to have continuous derivatives for optimisation 
	f_tanh(t) = (tanh(100*(t-160))/2 - tanh(100*(t-160.8))/2)
	@register_symbolic f_tanh(t)
	
end

# ╔═╡ 247533da-69ba-483c-8025-004c32a1bf9c
# Check the light signal functions 
begin
	x = LinRange(158, 163, 200)
	y1 = f.(x)
	y2 = f_tanh.(x)
	plot(x, [y1, y2], labels =["Step function" "Tanh"], lw = 2, title="Comparison of light signal functions")
end

# ╔═╡ 4b16437a-accf-4895-9b54-99736a75bd83
md"""
##### - Import RhoA data
"""

# ╔═╡ cd315a91-fd2c-4b93-a643-72c041b711d6
# Import data saved in txt files (Python preprocessing pickle file)
begin
	path_to_data = "../Data/"
	median_rhoa_FA = readdlm(Base.Filesystem.joinpath(path_to_data, "median_rhoa_FA_dynamics.txt"), ' ', Float64, '\n')[:, 2]
	median_rhoa_FA_KO = readdlm(Base.Filesystem.joinpath(path_to_data, "median_rhoa_FA_KO_dynamics.txt"), ' ', Float64, '\n')[:, 2]
	timeframe = LinRange(10, 400, 40)
end

# ╔═╡ a7e00912-b724-4ad4-85c7-aa1d740591d8
# Visualise data
begin
	plot(median_rhoa_FA_KO, label = "RhoA FA DLC1-KO", color="#800080")
	plot!(median_rhoa_FA, label = "RhoA FA DLC1", title= "RhoA dynamics at focal adhesions", color="#006400")
end

# ╔═╡ 55b92dd8-c651-48f5-be6d-30eebb723963
md"""
### RhoA data fitting
"""

# ╔═╡ 9db6fe47-0391-4e7f-84ae-ecf650f9418e
md"""
#### 1: focal adhesions and WT cells
"""

# ╔═╡ d352c2f2-1dee-429e-ba4e-1e8860c1f987
# Define RhoA negative autoregulation model for RhoA dynamics in WT
@mtkmodel Rho_dynamics begin
	
	@parameters begin
		Kon_GAP	
		Koff_GAP
		Kon_Rho
		Koff_Rho
	end
	
	@variables begin
		GEF(t), [bounds = (0, 10)]
		GAP(t), [bounds = (0, 10)]
		Rho(t), [bounds = (0, 10)]
		S(t)
	end
	
	@equations begin
		D(GEF) ~ 0.034 * (0.2 * S - GEF)
		D(GAP) ~ Kon_GAP * (1 - GAP) * (Rho-1) - Koff_GAP * GAP
		D(Rho) ~ Kon_Rho * GEF * (2 - Rho) - Koff_Rho * GAP * Rho
		S ~ f_tanh(t)
	end
end

# ╔═╡ eb81d398-90e5-4d4e-bab5-3df9ac5df099
@mtkbuild Rho_model = Rho_dynamics()

# ╔═╡ 05840396-3ee2-46d1-99aa-dda4c209a39f
begin

	# Define some initial parameters for the ODE model
	parameter_guess = [0.1, 0.12, 0.13, 0.14] 

	# Build ODE problems with guess parameters and initial conditions
	Rho_prob = ODEProblem(Rho_model, 
		[Rho_model.GEF => 0, 
			Rho_model.Rho => 1,
			Rho_model.GAP => 0
		], 
		(150.0, 400.0), 
		parameter_guess, jac = true)
end

# ╔═╡ bd48010a-96f9-4922-b86e-bacb1d474568
# Define MSE loss function to estimate parameters for WT RhoA data
function MSE_loss(new_parameters)

	# Substitute the initial conditions in the parameters
	new_u0 = [0; 0; 1]
	new_p = new_parameters[1:4]
	
	# Update the ODE problem with the new set of parameters
	new_ODE_prob = remake(Rho_prob, p = new_p, u0 = new_u0)

	# Solve the new ODE problem (saveat = numerical estimate time interval)
	new_sol = solve(new_ODE_prob, AutoVern9(Rodas4P()), saveat = 10, verbose = false, maxiters = 1e7, dtmax=0.05)
	new_sol_array = new_sol[Rho_model.Rho]

	# Determine the loss by computing the sum of squares
	if length(new_sol_array) == length(median_rhoa_FA[15:end])
		loss = sum(abs2, new_sol_array .- median_rhoa_FA[15:end])
	else
		loss = 1000
	end

	return loss
end

# ╔═╡ 4b34bb5f-1be5-47a1-a2d7-677a86e7503b
begin
	# Define optimization function and differentiation method
	optimisation_f = Optimization.OptimizationFunction((x, p) -> MSE_loss(x), Optimization.AutoForwardDiff())

	# Keep track of loss values during iterations
	track_loss = []

	# Set upper and lower bound for the parameters and initial conditions
	lower_b = repeat([0.0001], 4)
	upper_b = repeat([5], 4)

	# Define the optimisation problem
	optimisation_prob = Optimization.OptimizationProblem(optimisation_f, parameter_guess, lb = lower_b, ub=upper_b)

end

# ╔═╡ fe29f3cc-2f90-4c33-b70f-d8ba91e950ea
# Define callback function that is run at each iteration
callback! = function(parameters, loss)

	# Add current loss value to tracking array
	push!(track_loss, loss)
	
	# Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
	return false
end

# ╔═╡ b2400664-474d-4a16-90e5-7c789576c2cd
# ╠═╡ disabled = true
#=╠═╡
begin 

	# Clear the tracking array if not empty
	if !isempty(track_loss)
		empty!(track_loss)
	end
	
	# Run the optimisation
	optim_results = Optimization.solve(optimisation_prob, LBFGS(), callback = callback!, maxiters = 100, progress = true)
end
  ╠═╡ =#

# ╔═╡ f31fa805-d0f8-4f04-abaf-d17e0483350e
#=╠═╡
begin
	# Define initial conditions
	pred_u0 = [0; 0; 1]

	# Re-modify parameters as for optimisation
	pred_p = optim_results[1:4]

	# Solve the ODE problem with optimal parameters
	pred_sol = solve(remake(Rho_prob, p=pred_p, tspan=(120, 400)), u0=pred_u0, saveat=0.5, dtmax = 0.05)

	# Plot predicted model compared to data
	scatter(timeframe[12:end], median_rhoa_FA[12:end], label = "Data", lw=2, title = "FA RhoA dynamics in WT cells")
	plot!(pred_sol, label = "Prediction", lw=2, idxs=3, ylabel = "Fold-change \nnormalised to baseline", xlabel = "Time [s]")
end
  ╠═╡ =#

# ╔═╡ 619564f9-fafd-4750-bc5d-95d3c234e4b1
#=╠═╡
# Visualise optimisation statistics
optim_results.original
  ╠═╡ =#

# ╔═╡ ffbfc081-1ed0-4402-9174-cfbef1bb9fb3
# Plot the loss value at each iteration
plot(track_loss, title = "Evolution of the loss value", label = false)

# ╔═╡ 21f8b5e7-056e-4ea6-abc5-d23d7c8bddfe
#=╠═╡
plot(pred_sol, title = "aGEF, aGAP and aRhoA simultaneous dynamics", xlabel = "Time [s]")
  ╠═╡ =#

# ╔═╡ ceea52cf-7b52-44a1-af60-127c0eb76bde
begin
	# Save best parameters
	saved_parameters = [0.165441, 5.0, 0.00123041, 1.28402]

	# Transform back ODE parameters to their symbolic expression
	saved_param = [
		Rho_model.Kon_GAP => saved_parameters[get_idxparam(Rho_model.Kon_GAP, Rho_prob)],
		Rho_model.Koff_GAP => saved_parameters[get_idxparam(Rho_model.Koff_GAP, Rho_prob)],
		Rho_model.Kon_Rho => saved_parameters[get_idxparam(Rho_model.Kon_Rho, Rho_prob)],
		Rho_model.Koff_Rho => saved_parameters[get_idxparam(Rho_model.Koff_Rho, Rho_prob)]]
	
	saved_u0 = [0;0;1]

	# Solve the ODE with the best parameters and plot
	saved_pred_sol = solve(remake(Rho_prob, p=saved_param, u0=saved_u0), saveat=1, dtmax=0.05, tspan=(120, 400)) 
	scatter(timeframe[12:40], median_rhoa_FA[12:40], label = "Data", lw=2, title = "FA RhoA dynamics in WT cells")
	plot!(saved_pred_sol, label = "Prediction", lw=2, idxs=3, ylabel = "Fold-change \nnormalised to baseline", xlabel = "Time [s]")
	
end

# ╔═╡ 07cf038c-1032-4851-bdb1-9c17f290062b
saved_param

# ╔═╡ 91954a15-bf7c-40cf-a485-1db4bcae08cc
md"""
#### 2: focal adhesions and DLC1-KO cells
"""

# ╔═╡ a41bffdf-5564-4e08-a99f-d1de48985584
# Define RhoA negative autoregulation model for DLC1-KO cells
@mtkmodel Rho_KO_dynamics begin
	
	@parameters begin
		Kon_GAP
		Koff_GAP
		Kon_Rho
		Koff_Rho
	end
	
	@variables begin
		GEF(t), [bounds = (0, 10)]
		GAP(t), [bounds = (0, 10)]
		Rho(t), [bounds = (0, 10)]
		S(t)
	end
	
	@equations begin
		D(GEF) ~  0.034 * (0.2 * S - GEF)
		D(GAP) ~ Kon_GAP * (1 - GAP) * (Rho-1)^2 * ((tanh(100*(Rho-1))-tanh(100*(Rho-2)))/2) - Koff_GAP * GAP
		D(Rho) ~ Kon_Rho * GEF * (2 - Rho) - Koff_Rho * GAP^2 * Rho
		S ~ f_tanh(t)
	end
end

# ╔═╡ 1e9a10aa-f14d-4b35-b295-99198d9b835d
@mtkbuild Rho_KO_model = Rho_KO_dynamics()

# ╔═╡ 51218007-1366-4efc-be76-8596f1a6384c
begin

	# Define some initial parameters for the ODE model
	KO_parameter_guess = [0.1, 0.12, 0.13, 0.14]

	# Build ODE problems with guess parameters and initial conditions
	Rho_KO_prob = ODEProblem(Rho_KO_model, 
		[Rho_KO_model.GEF => 0, 
			Rho_KO_model.Rho => 1, 
			Rho_KO_model.GAP => 0
		], 
		(150.0, 400.0), 
		KO_parameter_guess, jac = true)
end

# ╔═╡ ef4cc0e0-6473-4a87-9f87-8f3c9a65e57f
# Define MSE loss function to estimate parameters
function KO_MSE_loss(new_parameters)

	# Substitute the initial conditions in the parameters
	new_u0 = [0;0;1]
	new_p = new_parameters[1:4]
	
	# Update the ODE problem with the new set of parameters
	KO_new_ODE_prob = remake(Rho_KO_prob, p=new_p, u0=new_u0)

	# Solve the new ODE problem (saveat = numerical estimate time interval)
	KO_new_sol = solve(KO_new_ODE_prob, AutoVern9(Rodas4P()), saveat = 10, maxiters = 1e7, verbose = false, dtmax=0.05)
	KO_new_sol_array = KO_new_sol[Rho_KO_model.Rho]

	# Determine the loss (sum of squares)
	if length(KO_new_sol_array) == length(median_rhoa_FA_KO[15:end])
		loss = sum(abs2, KO_new_sol_array .- median_rhoa_FA_KO[15:end])
	else
		loss = 1000
	end

	return loss
end

# ╔═╡ 0a3ec77c-36f7-4571-b9bc-7f64c45c393c
begin
	# Define optimization function and differentiation method
	KO_optimisation_f = Optimization.OptimizationFunction((x, p) -> KO_MSE_loss(x), Optimization.AutoForwardDiff())

	# Keep track of loss values during iterations
	KO_track_loss = []

	# Set upper and lower bound for the parameters and initial conditions
	KO_lower_b = repeat([0.001], 4)
	KO_upper_b = repeat([5], 4)

	# Define the optimisation problem
	KO_optimisation_prob = Optimization.OptimizationProblem(KO_optimisation_f, KO_parameter_guess, lb= KO_lower_b, ub=KO_upper_b)
end

# ╔═╡ d0af947c-168f-4ea1-aac9-b0117277b1aa
# Define callback function that is run at each iteration
KO_callback! = function(parameters, loss)

	# Add current loss value and parameter to tracking arrays
	push!(KO_track_loss, loss)
	
	# Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
	return false
end

# ╔═╡ 38a5bd18-9074-4830-b8e0-9e74e1650547
# ╠═╡ disabled = true
#=╠═╡
begin 
	# Clear the tracking array if not empty
	if length(KO_track_loss) > 0
		deleteat!(KO_track_loss, 1:length(KO_track_loss))
	end

	# Run the optimisation
	KO_optim_results = Optimization.solve(KO_optimisation_prob, LBFGS(), callback = KO_callback!, maxiters = 100, progress = true)
end
  ╠═╡ =#

# ╔═╡ 9fcdd6d4-5d71-4f4f-809a-09d76e34a09f
#=╠═╡
begin
	# Retrieve initial conditions from optimisation solution
	KO_predicted_u0 = [0;0;1]

	# Re-modify parameters as for optimisation
	KO_pred_p = KO_optim_results[1:4]

	# Solve the ODE problem with optimal parameters
	KO_pred_sol = solve(remake(Rho_KO_prob, p=KO_pred_p, u0=KO_predicted_u0, tspan=(120, 400)), saveat=0.5, dtmax = 0.05)

	scatter(timeframe[12:end], median_rhoa_FA_KO[12:end], label = "Data", lw=2, title = "RhoA dynamics in FA and DLC1-KO cells")
	plot!(KO_pred_sol, label = "Prediction", lw=2, idxs=3)

end
  ╠═╡ =#

# ╔═╡ 5b4ccf8e-e412-45eb-8823-b8fefdaaf2d9
#=╠═╡
KO_optim_results.original
  ╠═╡ =#

# ╔═╡ e8b4cb6f-8b04-49f1-8423-ab5479bf8e0a
# Plot the loss value at each iteration
plot(KO_track_loss, title = "Evolution of the loss value", label = false)

# ╔═╡ 8fda161f-4c5d-44cb-9e78-789495a6d0b1
#=╠═╡
plot(KO_pred_sol, title = "aGEF, aGAP and aRhoA simultaneous dynamics", xlabel = "Time [s]")
  ╠═╡ =#

# ╔═╡ f5877b34-cefe-4653-acd1-7ae9344e7d4f
begin
	# Save best parameters
	saved_KO_parameters = [0.0129794, 0.0261144, 5.0, 1.3195]

	# Transform back ODE parameters to their symbolic expression
	saved_KO_param = [
		Rho_KO_model.Kon_GAP => saved_KO_parameters[get_idxparam(Rho_KO_model.Kon_GAP, Rho_KO_prob)],
		Rho_KO_model.Koff_GAP => saved_KO_parameters[get_idxparam(Rho_KO_model.Koff_GAP, Rho_KO_prob)],
		Rho_KO_model.Kon_Rho => saved_KO_parameters[get_idxparam(Rho_KO_model.Kon_Rho, Rho_KO_prob)],
		Rho_KO_model.Koff_Rho => saved_KO_parameters[get_idxparam(Rho_KO_model.Koff_Rho, Rho_KO_prob)]]
	
	saved_KO_u0 = [0;0;1]

	# Solve the ODE with the best parameters and plot
	saved_KO_pred_sol = solve(remake(Rho_KO_prob, p=saved_KO_param, u0 = saved_KO_u0, tspan=(120, 400)), saveat=1, dtmax=0.05) 
	scatter(timeframe[12:end], median_rhoa_FA_KO[12:end], label = "Data", lw=2, title = "FA RhoA dynamics in WT cells")
	plot!(saved_KO_pred_sol, label = "Prediction", lw=2, idxs=3, ylabel = "Fold-change \nnormalised to baseline", xlabel = "Time [s]")
end

# ╔═╡ 837c314e-ea7b-4dab-8240-13da3a17d6e5
saved_KO_param

# ╔═╡ af2bb913-c1a6-489f-b5ae-3cece876e089
md"""
### Comparison between DLC1-KO and WT data fitting
"""

# ╔═╡ a4b2fb33-2b1c-4a8b-a8a7-69cedade9a6a
begin
	plot(saved_KO_pred_sol, lw=2, idxs=3, linecolor="#800080", label = "DLC1-KO (Prediction)", fontfamily="Arial")
	scatter!(timeframe[12:end], median_rhoa_FA_KO[12:end], label = "DLC1-KO (Data)", lw=2, title = "Focal Adhesions RhoA dynamics", markercolor="#800080")
	plot!(saved_pred_sol, lw=2, idxs=3, linecolor="#006400",label = "WT (Prediction)")
	scatter!(timeframe[12:end], median_rhoa_FA[12:end], label = "WT (Data)", markercolor="#006400")
	xlabel!("Time [s]")
	ylabel!("Fold-change \nnormalised to baseline")
	#savefig("../Data/FA_plot_linear.svg")
end

# ╔═╡ 91329d9a-3135-4499-ae5f-bfc909c0bc9e
begin
	gef_plot = plot(saved_pred_sol, title = "", xlabel = "", idxs=1, linecolor="#006400", linestyle=:dot, legend_font_pointsize=6, ylabel="relative [aGEF]", legend=:best, fontfamily="Arial")
	plot!(timeframe[12:end], NaN.*timeframe[12:end], label = "aGAP", linestyle=:dash, linecolor="#006400")
	plot!(twinx(), saved_pred_sol, title = "\nWT", legend=false, xlabel = "", idxs=2, linecolor="#006400", linestyle=:dash, titlefontsize=10, ylabel="relative [aGAP]")
	gef_plot_KO = plot(saved_KO_pred_sol, title = "DLC1-KO", xlabel = "Time [s]", idxs=1, linecolor="#800080", linestyle=:dot, titlefontsize=10, legend_font_pointsize=6, ylabel="relative [aGEF]", fontfamily="Arial")
	plot!(timeframe[12:end], NaN .* timeframe[12:end], label = "aGAP", linestyle=:dash, linecolor="#800080")
	plot!(twinx(), saved_KO_pred_sol, title = "", idxs=2, linecolor="#800080", linestyle=:dash, xlabel = "", legend=false, ylabel="relative [aGAP]")
	plot(gef_plot, gef_plot_KO, layout=(2, 1), plot_title="Focal Adhesions aGEF and aGAP dynamics", plot_titlelocation=:left, plot_titlefontsize=14, )
	#savefig("../Data/FA_all_plot_sharex_linear.svg")
end

# ╔═╡ da69d08a-701b-4fed-9643-42fcd4c11a2e
""" 
	get_idxparam(symbol::Symbol, prob::ODEProblem)

Custom function to retrieve the index of the parameter in the optimisation solution array using its symbolic expression (symbol). 
"""
function get_idxparam(symbol, prob)

	mtk_p = prob.ps
	mtk_p_array = prob.p.tunable
	for i in eachindex(mtk_p_array)
		if mtk_p[symbol] == mtk_p_array[i]
			return i
		end
	end
end

# ╔═╡ Cell order:
# ╟─de0e5ef4-c53d-4659-9bc0-73f7b10e923d
# ╟─cfa5e8b3-cfea-42bc-ab54-023d9de67850
# ╟─a97176f9-7dd2-4870-9a80-b91e25ba8b97
# ╟─44e0f32e-3a41-44d6-bf44-0dc7d323c0b6
# ╠═52bdbc46-53fc-42ca-8fec-e43522c30063
# ╟─247533da-69ba-483c-8025-004c32a1bf9c
# ╟─4b16437a-accf-4895-9b54-99736a75bd83
# ╟─cd315a91-fd2c-4b93-a643-72c041b711d6
# ╠═a7e00912-b724-4ad4-85c7-aa1d740591d8
# ╟─55b92dd8-c651-48f5-be6d-30eebb723963
# ╟─9db6fe47-0391-4e7f-84ae-ecf650f9418e
# ╠═d352c2f2-1dee-429e-ba4e-1e8860c1f987
# ╟─eb81d398-90e5-4d4e-bab5-3df9ac5df099
# ╠═05840396-3ee2-46d1-99aa-dda4c209a39f
# ╠═bd48010a-96f9-4922-b86e-bacb1d474568
# ╠═fe29f3cc-2f90-4c33-b70f-d8ba91e950ea
# ╠═4b34bb5f-1be5-47a1-a2d7-677a86e7503b
# ╠═b2400664-474d-4a16-90e5-7c789576c2cd
# ╠═f31fa805-d0f8-4f04-abaf-d17e0483350e
# ╟─619564f9-fafd-4750-bc5d-95d3c234e4b1
# ╟─ffbfc081-1ed0-4402-9174-cfbef1bb9fb3
# ╟─21f8b5e7-056e-4ea6-abc5-d23d7c8bddfe
# ╠═ceea52cf-7b52-44a1-af60-127c0eb76bde
# ╟─07cf038c-1032-4851-bdb1-9c17f290062b
# ╟─91954a15-bf7c-40cf-a485-1db4bcae08cc
# ╟─a41bffdf-5564-4e08-a99f-d1de48985584
# ╟─1e9a10aa-f14d-4b35-b295-99198d9b835d
# ╟─51218007-1366-4efc-be76-8596f1a6384c
# ╟─ef4cc0e0-6473-4a87-9f87-8f3c9a65e57f
# ╟─d0af947c-168f-4ea1-aac9-b0117277b1aa
# ╟─0a3ec77c-36f7-4571-b9bc-7f64c45c393c
# ╠═38a5bd18-9074-4830-b8e0-9e74e1650547
# ╟─9fcdd6d4-5d71-4f4f-809a-09d76e34a09f
# ╟─5b4ccf8e-e412-45eb-8823-b8fefdaaf2d9
# ╟─e8b4cb6f-8b04-49f1-8423-ab5479bf8e0a
# ╟─8fda161f-4c5d-44cb-9e78-789495a6d0b1
# ╠═f5877b34-cefe-4653-acd1-7ae9344e7d4f
# ╟─837c314e-ea7b-4dab-8240-13da3a17d6e5
# ╟─af2bb913-c1a6-489f-b5ae-3cece876e089
# ╟─a4b2fb33-2b1c-4a8b-a8a7-69cedade9a6a
# ╟─91329d9a-3135-4499-ae5f-bfc909c0bc9e
# ╟─da69d08a-701b-4fed-9643-42fcd4c11a2e
