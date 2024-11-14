### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 8277b2e6-6cf6-4f4a-ae4b-274548edb651
begin
	import Pkg
	Pkg.activate(".")
	using OrdinaryDiffEq, Optimization, OptimizationPolyalgorithms, SciMLSensitivity, ModelingToolkit, Optim, ForwardDiff, Plots, DelimitedFiles
end

# ╔═╡ ffd55ae0-dbed-474b-9b89-10bcf3bdcfe8
md"""
### Environment setup
"""

# ╔═╡ a97176f9-7dd2-4870-9a80-b91e25ba8b97
md"""
## Data fitting of RhoA activation/deactivation dynamics at the membrane
This notebook contains the code used to fit an ODE model to the experimental data of RhoA activity obtained after local light stimulation at focal adhesions as described in the paper by Heydasch et al. \


Max Heydasch, Lucien Hinderling, Jakobus van Unen, Maciej Dobrzynski, Olivier Pertz
(2023) \
**GTPase activating protein DLC1 spatio-temporally regulates Rho signaling** \
eLife 12:RP90305 \
https://doi.org/10.7554/eLife.90305.1  

Data source: Lucien Hinderling, PertzLab
"""

# ╔═╡ e44a2ff8-7326-4080-813e-ffbeb4c4b648
md"""
### Initial steps
##### - Light signal function
To use the gradient for parameter optimisation the ODE system needs to be composed of continuous and differentiable function to avoid instability with the ODE solver. Instead of a step function, the light signal is modelled using a hyperbolic tangent.
"""

# ╔═╡ 52bdbc46-53fc-42ca-8fec-e43522c30063
# Define symbolic functions and variables for ODE model
begin
	@independent_variables t
	D = Differential(t)
	
	# Light signal function (step function)
	f(t) = (160 <= t <= 160.8)
	@register_symbolic f(t) 

	# Use hyperbolic tangent to have continuous derivatives for data fitting as light signal function
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

# ╔═╡ b05d3803-9883-4073-87a8-6324db7c207f
md"""
##### - Import RhoA data
"""

# ╔═╡ a20dcc69-c419-472d-8b93-836cde4a54e7
# Import data saved in txt files (Python preprocessing pickle file)
begin
	path_to_data = "../Data/"
	median_rhoa_NONFA = readdlm(Base.Filesystem.joinpath(path_to_data, "median_rhoa_NONFA_dynamics.txt"), ' ', Float64, '\n')[:, 2]
	median_rhoa_NONFA_KO = readdlm(Base.Filesystem.joinpath(path_to_data, "median_rhoa_NONFA_KO_dynamics.txt"), ' ', Float64, '\n')[:, 2]
	timeframe = LinRange(10, 400, 40)
end

# ╔═╡ b7208c41-3f65-4f6a-ae06-e4426d6549a0
# Visualise data
begin
	plot(median_rhoa_NONFA_KO, label = "RhoA DLC1-KO", fontfamily="Arial", color = "#800080")
	plot!(median_rhoa_NONFA, label = "RhoA DLC1", title= "RhoA dynamics at membrane", color="#006400")
end

# ╔═╡ 487b4ff4-3e29-48d0-bb6d-9d9ec048de1c
md"""
### Data fitting
"""

# ╔═╡ f025a699-1e30-478c-8160-4ea16005eac7
md"""
#### 1: membrane and WT cells
"""

# ╔═╡ c654e848-5777-4443-87b4-58e48b353f01
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
		# Define linear model with Km = 0
		D(GEF) ~  0.034 * (0.2 * S - GEF)
		D(GAP) ~ Kon_GAP * (1 - GAP) * (Rho - 1) - Koff_GAP * GAP
		D(Rho) ~  Kon_Rho * GEF * (2 - Rho) - Koff_Rho * GAP * Rho
		S ~ f_tanh(t)
	end
end

# ╔═╡ f6eef16c-6084-42ee-80ea-fc4165b78753
@mtkbuild Rho_model = Rho_dynamics()

# ╔═╡ 979b822d-cbc4-49b5-b2d4-8821fc664917
begin

	# Define some initial parameters for the ODE model
	parameter_guess = [0.1, 0.12, 0.13, 0.14]

	# Build ODE problems with guess parameters and initial conditions
	Rho_prob = ODEProblem(Rho_model, 
		[Rho_model.GEF => 0, 
			Rho_model.Rho => 1,
			Rho_model.GAP => 0], 
		(150.0, 400.0), parameter_guess, jac = true)
end

# ╔═╡ 3fae1a20-4231-4758-a403-11e929d36a10
# Define MSE loss function to estimate parameters
function MSE_loss(new_parameters)
	
	# Substitute the initial conditions in the parameters
	new_u0 = [0;0;1]
	new_p = new_parameters[1:4]

	# Update the ODE problem with the new set of parameters
	new_ODE_prob = remake(Rho_prob, p=new_p, u0=new_u0)

	# Solve the new ODE problem (saveat = numerical estimate time interval)
	new_sol = solve(new_ODE_prob, AutoVern9(Rodas4P()), saveat = 10, maxiters = 1e7, dtmax = 0.05, verbose = false)
	new_sol_array = new_sol[Rho_model.Rho]

	# Determine the loss (sum of squares)
	if length(new_sol_array) == length(median_rhoa_NONFA[15:end])
		loss = sum(abs2, new_sol_array .- median_rhoa_NONFA[15:end])
	else
		loss = 1000
	end

	return loss#, new_sol
end

# ╔═╡ 5d479288-888c-4a32-8215-e5cf6b0713ca
begin
	# Define optimization function and differentiation method
	optimisation_f = Optimization.OptimizationFunction((x, p) -> MSE_loss(x), Optimization.AutoForwardDiff())

	# Keep track of loss values during iterations
	track_loss = []

	# Define upper and lower bound for parameters and initial conditions
	lower_b = repeat([0.001], 4)
	upper_b = repeat([5], 4)
	
	# Define the optimisation problem
	optimisation_prob = Optimization.OptimizationProblem(optimisation_f, parameter_guess, lb = lower_b, ub = upper_b)

end

# ╔═╡ 4640701b-a251-46f0-b94c-b345b46e8ab1
# Define callback function that is run at each iteration
callback! = function(parameters, loss)

	# Add current loss value and parameter to tracking arrays
	push!(track_loss, loss)
	
	# Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
	return false
end

# ╔═╡ e4389a99-b2e8-4938-843d-343f9dd3df5f
begin 

	# Clear the tracking arrays if not empty
	if length(track_loss) > 0
		deleteat!(track_loss, 1:length(track_loss))
	end

	# Run the optimisation
	optim_results = Optimization.solve(optimisation_prob, LBFGS(), callback = callback!, maxiters = 100, progress = true)
end

# ╔═╡ 16fc1498-503b-4447-9ea9-82879efdf3f8
begin
	# Retrieve initial conditions from optimisation solution
	pred_u0 = [0; 0; 1]

	# Re-modify parameters as for optimisation
	pred_p = optim_results[1:4]

	# Solve the ODE problem with optimal parameters
	pred_sol = solve(remake(Rho_prob, p=pred_p, u0=pred_u0, tspan=(120, 400)), saveat=1, dtmax=0.05)
	scatter(timeframe[12:end], median_rhoa_NONFA[12:end], label = "Data", lw=2, title = "Membrane RhoA dynamics in WT cells")
	plot!(pred_sol, label = "Prediction", lw=2, idxs=3, ylabel = "Fold-change \nnormalised to baseline", xlabel = "Time [s]")
end

# ╔═╡ 476286e8-dd0d-4036-b3ab-e9f075ce8205
# Visualise optimisation statistics
optim_results.original

# ╔═╡ 2efefa13-fad1-4ad1-8265-32cf46ffddbb
# Plot the loss value at each iteration
plot(track_loss, title = "Evolution of the loss value", label = false)

# ╔═╡ fab82aa4-a281-4bb9-8771-378cef7bf613
begin
	# Plot the dynamics of all proteins
	plot(pred_sol, title = "aGEF, aGAP and aRhoA simultaneous dynamics", xlabel = "Time [s]")
end

# ╔═╡ 5948c768-4b3c-4bb2-b880-6723fafc9053
md"""
#### 2: membrane and DLC1-KO cells
"""

# ╔═╡ df9db3fd-2e86-496a-afc5-f27887cdda12
# Define RhoA negative autoregulation model for membrane and DLC1-KO cells
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

		# Define linear model
		D(GEF) ~ 0.034 * (0.2 * S - GEF)
		D(GAP) ~ Kon_GAP * (1 - GAP) * (Rho-1)^2 - Koff_GAP * GAP
		D(Rho) ~ Kon_Rho * GEF * (2 - Rho) - Koff_Rho * (GAP)^2 * Rho
		S ~ f_tanh(t)
	end
end

# ╔═╡ b6f6741b-53cf-4e24-8017-dac404d0a04d
@mtkbuild Rho_KO_model = Rho_KO_dynamics()

# ╔═╡ fe32650c-3795-4c93-a04c-a3828241f85e
begin

	# Define some initial parameters for the ODE model
	KO_parameter_guess = [0.1, 0.2, 0.3, 0.4]
	
	# Build ODE problems with guess parameters and initial conditions
	Rho_KO_prob = ODEProblem(Rho_KO_model, 
		[Rho_KO_model.GEF => 0., 
			Rho_KO_model.Rho => 1.0, 
			Rho_KO_model.GAP => 0.], 
		(150.0, 400.0), KO_parameter_guess, jac = true)
end

# ╔═╡ 89b95eca-fef6-43b9-bacc-401aabf075d4
# Define MSE loss function to estimate parameters
function KO_MSE_loss(new_parameters)
	
	# Substitute the initial conditions in the parameters
	new_u0 = [0;0;1]
	new_p = new_parameters[1:4]
	
	# Update the ODE problem with the new set of parameters
	KO_new_ODE_prob = remake(Rho_KO_prob, p=new_p, u0 = new_u0)

	# Solve the new ODE problem (saveat = numerical estimate time interval)
	KO_new_sol = solve(KO_new_ODE_prob, AutoVern9(Rodas4P()), saveat = 10, maxiters = 1e7, dtmax = 0.05, verbose = false)
	KO_new_sol_array = KO_new_sol[Rho_KO_model.Rho]
	
	# Compute the loss (sum of squares)
	if length(KO_new_sol_array) == length(median_rhoa_NONFA_KO[15:end])
		loss = sum(abs2, KO_new_sol_array .- median_rhoa_NONFA_KO[15:end])

	else
		loss = 1000
	end

	return loss
end

# ╔═╡ 1148d27b-d16a-41c7-9795-81331f1d32f0
begin
	# Define optimization function and differentiation method
	KO_optimisation_f = Optimization.OptimizationFunction((x, p) -> KO_MSE_loss(x), Optimization.AutoForwardDiff())

	# Keep track of loss values during iterations
	KO_track_loss = []

	# Define upper and lower bound for parameters and initial conditions
	KO_lower_b = repeat([0.001], 4)
	KO_upper_b = repeat([5], 4)
	
	# Define the optimisation problem
	KO_optimisation_prob = Optimization.OptimizationProblem(KO_optimisation_f, KO_parameter_guess, ub = KO_upper_b, lb = KO_lower_b)
end

# ╔═╡ 764df649-41df-417e-bfc9-19376333550d
# Define callback function that is run at each iteration
KO_callback! = function(parameters, loss)

	# Add current loss value to tracking array
	push!(KO_track_loss, loss)
	
	# Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
	return false
end

# ╔═╡ 13c1ea9b-9804-4f9c-8429-6c024284bf8e
begin 
	# Clear the tracking array if not empty
	if !isempty(KO_track_loss)
		empty!(KO_track_loss)
	end

	# Run the optimisation
	KO_optim_results = Optimization.solve(KO_optimisation_prob, LBFGS(), callback = KO_callback!, maxiters = 100, progress = true)
end

# ╔═╡ c6bfd2a4-df9d-4f83-882d-d1bb1c1a3ecb
begin
	# Retrieve initial conditions from optimisation solution
	KO_pred_u0 = [0;0;1]

	# Re-modify parameters as for optimisation
	KO_pred_p = KO_optim_results[1:4]

	# Solve and plot final solution
	KO_pred_sol = solve(remake(Rho_KO_prob, p=KO_pred_p, u0=KO_pred_u0, tspan=(120, 400)), saveat=1, dtmax=0.01)
	scatter(timeframe[12:end], median_rhoa_NONFA_KO[12:end], label = "Data", lw=2, title = "Membrane RhoA dynamics in DLC1-KO cells" )
	plot!(KO_pred_sol, label = "Prediction", lw=2, idxs=3, ylabel = "Fold-change \nnormalised to baseline", xlabel = "Time [s]")
end

# ╔═╡ 09cce5a5-713d-445c-aadf-34f9a820be56
# Visualise optimisation statistics
KO_optim_results.original

# ╔═╡ ea8d3c39-e086-4db0-b50f-f1b64bc992cf
begin
	# Plot the dynamics of all proteins
	plot(KO_pred_sol, title = "aGEF, aGAP and aRhoA simultaneous dynamics", xlabel = "Time [s]")
end

# ╔═╡ 0763ed96-04bb-43b3-8c40-fa58344ea96f
md"""
### Comparison between DLC1-KO and WT data fitting
"""

# ╔═╡ f58afeb2-8704-4d1a-9f12-dd65ab05dc87
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

# ╔═╡ 4d6ab3c4-d077-4451-a96b-6d8d7c8e1beb
begin
	# Save best parameters
	#saved_parameters = [0.0323253, 5.0, 1.22424, 0.0213587, 0.0213584, 5.0, 0.2448, 1.0]
	#saved_parameters = [1.28385e-9, 4.08817, 0.28807, 0.045259, 0.0491, 3.84274, 0.0710608, 0.877017]
	#saved_parameters = [2.51715e-22, 5.14643, 0.0526724, 10.0, 2.22239e-5, 4.87384, 0.389004, 3.86332e-15]
	#saved_parameters = [0.0336366, 0.725186, 0.0300675, 0.0102179, 0.00439217, 4.82675, 1.00399, 1.00399] # no GAP/GEF steady states and deactivation of RhoA depends on unnormalised RhoA
	#saved_parameters = [0.0298006, 0.0228549, 0.00293089, 0.0139713] # by setting 1 as max rate
	#saved_parameters = [0.0452812, 2.22172e-22, 0.000153249, 1.10432] # with 0.2 as GEF0 and 0.12 as Kcatgef and 3 as max rate
	#saved_parameters = [0.308185, 0.000165521, 9.41079] # with 0.2 as GEF0 and 0.12 as Kcatgef and 10 as max rate
	#saved_parameters = [0.0250921, 0.00229808, 0.0147689] # with 0.2 as GEF0 and 0.1 as Kcatgef and 10 as max rate
	saved_parameters = [0.155982, 0.00019739, 2.97246] # with 0.2 as GEF0 and 0.14 as Kcatgef and 3 as max rate
	#saved_parameters = [0.0457636, 0.000155832, 1.06384] # with 0.2 as GEF0 and 0.12 as Kcatgef and 3 as max rate
	#saved_parameters = [0.0301377, 0.107515, 0.00469657, 0.00869423] # with 0.2 as GEF0 and optim as Kcatgef and 3 as max rate
	saved_parameters = [0.118373, 5.0, 0.00135633, 1.22779]
	
	# Transform back ODE parameters to their symbolic expression
	saved_param = [
		Rho_model.Kon_GAP => saved_parameters[get_idxparam(Rho_model.Kon_GAP, Rho_prob)],
		Rho_model.Koff_GAP => saved_parameters[get_idxparam(Rho_model.Koff_GAP, Rho_prob)],
		Rho_model.Kon_Rho => saved_parameters[get_idxparam(Rho_model.Kon_Rho, Rho_prob)],
		Rho_model.Koff_Rho => saved_parameters[get_idxparam(Rho_model.Koff_Rho, Rho_prob)]
	]
	saved_u0 = [0;0;1]

	# Solve the ODE with the best parameters and plot
	saved_pred_sol = solve(remake(Rho_prob, p=saved_param, u0=saved_u0, tspan=(120, 400)), saveat=1, dtmax=0.05) 
	scatter(timeframe[12:end], median_rhoa_NONFA[12:end], label = "Data", lw=2, title = "Membrane RhoA dynamics in WT cells")
	plot!(saved_pred_sol, label = "Prediction", lw=2, idxs=3, ylabel = "Fold-change \nnormalised to baseline", xlabel = "Time [s]")
end

# ╔═╡ 50000f94-3949-4f4d-911d-019c4d1a3115
saved_param

# ╔═╡ d758e312-70d8-41e3-a1dc-cc6ce8a744db
begin
	# Save best parameters
	#saved_KO_parameters = [0.0153623, 5.0, 3.34273, 0.0419158, 5.0, 0.320935, 0.999999] 
	#saved_KO_parameters = [0.0142832, 0.938937, 3.0, 0.00409945] # with 3 as max rate, 0.94 as gef0 and 0.12 as kcatgef
	#saved_KO_parameters = [0.014226, 10.0, 0.00225018] # with 10 as max rate, 0.2 as gef0 and 0.14 as kcatgef
	#saved_KO_parameters = [0.0152448, 0.00487144, 0.0819561] # with 3 as max rate, 0.2 as gef0 and 0.14 as kcatgef
	saved_KO_parameters = [0.016281, 3.0, 0.00312485] # with 3 as max rate, 0.2 as gef0 and 0.15 as kcatgef
	#saved_KO_parameters = [1.30773, 3.0, 0.415258, 3.0] # with 3 as max rate, 0.2 as gef0 and optim as kcatgef
	saved_KO_parameters = [0.0138277, 0.0315648, 5.0, 1.36068]

	# Transform back ODE parameters to their symbolic expression
	saved_KO_param = [
		Rho_KO_model.Kon_GAP => saved_KO_parameters[get_idxparam(Rho_KO_model.Kon_GAP, Rho_KO_prob)],
		Rho_KO_model.Koff_GAP => saved_KO_parameters[get_idxparam(Rho_KO_model.Koff_GAP, Rho_KO_prob)],
		Rho_KO_model.Kon_Rho => saved_KO_parameters[get_idxparam(Rho_KO_model.Kon_Rho, Rho_KO_prob)], 
		Rho_KO_model.Koff_Rho => saved_KO_parameters[get_idxparam(Rho_KO_model.Koff_Rho, Rho_KO_prob)]]
	saved_KO_u0 = [0;0;1] 

	# Solve the ODE with the best parameters and plot
	saved_KO_pred_sol = solve(remake(Rho_KO_prob, p=saved_KO_param, u0=saved_KO_u0, tspan=(120, 400)), saveat=1, dtmax=0.05) 
	scatter(timeframe[12:end], median_rhoa_NONFA_KO[12:end], label = "Data", lw=2, title = "Membrane RhoA dynamics in DLC1-KO cells")
	plot!(saved_KO_pred_sol, label = "Prediction", lw=2, idxs=3, ylabel = "Fold-change \nnormalised to baseline", xlabel = "Time [s]")
end

# ╔═╡ ee1b032c-e614-4251-90a7-4f50da3b61d1
saved_KO_param

# ╔═╡ 33ec1636-17f6-4e89-a42f-e368df94b965
begin
	# Plot results for each condition (WT and DLC1-KO)
	plot(saved_KO_pred_sol, lw=2, idxs=3, linecolor="#800080", label = "DLC1-KO (Prediction)", fontfamily="Arial")
	scatter!(timeframe[12:end], median_rhoa_NONFA_KO[12:40], label = "DLC1-KO (Data)", lw=2, title = "Membrane RhoA dynamics", markercolor="#800080")
	plot!(saved_pred_sol, lw=2, idxs=3, linecolor="#006400",label = "WT (Prediction)")
	scatter!(timeframe[12:end], median_rhoa_NONFA[12:end], label = "WT (Data)", markercolor="#006400")
	xlabel!("Time [s]")
	ylabel!("Fold-change \nnormalised to baseline")
	#savefig("../Data/membrane_plot_linear.svg")
end

# ╔═╡ 65562ef7-93eb-4d05-98fe-6689c26c4063
begin
	gef_plot = plot(saved_pred_sol, title = "", xlabel = "", idxs=1, linecolor="#006400", linestyle=:dot, legend_font_pointsize=6, ylabel="relative [aGEF]", legend=:best, fontfamily="Arial")
	plot!(timeframe[12:end], NaN.*timeframe[12:end], label = "aGAP", linestyle=:dash, linecolor="#006400")
	plot!(twinx(), saved_pred_sol, title = "\nWT", legend=false, xlabel = "", idxs=2, linecolor="#006400", linestyle=:dash, titlefontsize=10, ylabel="relative [aGAP]")
	gef_plot_KO = plot(saved_KO_pred_sol, title = "DLC1-KO", xlabel = "Time [s]", idxs=1, linecolor="#800080", linestyle=:dot, titlefontsize=10, legend_font_pointsize=6, ylabel="relative [aGEF]", fontfamily="Arial")
	plot!(timeframe[12:end], NaN.*timeframe[12:end], label = "aGAP", linestyle=:dash, linecolor="#800080")
	plot!(twinx(), saved_KO_pred_sol, title = "", idxs=2, linecolor="#800080", linestyle=:dash, xlabel = "", legend=false, ylabel="relative [aGAP]")
	plot(gef_plot, gef_plot_KO, layout=(2, 1), plot_title="Membrane aGEF and aGAP dynamics", plot_titlelocation=:left, plot_titlefontsize=14, )
	#savefig("../Data/membrane_all_plot_linear.svg")
end

# ╔═╡ Cell order:
# ╟─ffd55ae0-dbed-474b-9b89-10bcf3bdcfe8
# ╠═8277b2e6-6cf6-4f4a-ae4b-274548edb651
# ╟─a97176f9-7dd2-4870-9a80-b91e25ba8b97
# ╟─e44a2ff8-7326-4080-813e-ffbeb4c4b648
# ╠═52bdbc46-53fc-42ca-8fec-e43522c30063
# ╟─247533da-69ba-483c-8025-004c32a1bf9c
# ╟─b05d3803-9883-4073-87a8-6324db7c207f
# ╠═a20dcc69-c419-472d-8b93-836cde4a54e7
# ╠═b7208c41-3f65-4f6a-ae06-e4426d6549a0
# ╟─487b4ff4-3e29-48d0-bb6d-9d9ec048de1c
# ╟─f025a699-1e30-478c-8160-4ea16005eac7
# ╠═c654e848-5777-4443-87b4-58e48b353f01
# ╟─f6eef16c-6084-42ee-80ea-fc4165b78753
# ╠═979b822d-cbc4-49b5-b2d4-8821fc664917
# ╠═3fae1a20-4231-4758-a403-11e929d36a10
# ╠═4640701b-a251-46f0-b94c-b345b46e8ab1
# ╠═5d479288-888c-4a32-8215-e5cf6b0713ca
# ╠═e4389a99-b2e8-4938-843d-343f9dd3df5f
# ╠═16fc1498-503b-4447-9ea9-82879efdf3f8
# ╠═476286e8-dd0d-4036-b3ab-e9f075ce8205
# ╠═2efefa13-fad1-4ad1-8265-32cf46ffddbb
# ╠═fab82aa4-a281-4bb9-8771-378cef7bf613
# ╠═4d6ab3c4-d077-4451-a96b-6d8d7c8e1beb
# ╟─50000f94-3949-4f4d-911d-019c4d1a3115
# ╟─5948c768-4b3c-4bb2-b880-6723fafc9053
# ╠═df9db3fd-2e86-496a-afc5-f27887cdda12
# ╟─b6f6741b-53cf-4e24-8017-dac404d0a04d
# ╠═fe32650c-3795-4c93-a04c-a3828241f85e
# ╠═89b95eca-fef6-43b9-bacc-401aabf075d4
# ╠═764df649-41df-417e-bfc9-19376333550d
# ╠═1148d27b-d16a-41c7-9795-81331f1d32f0
# ╠═13c1ea9b-9804-4f9c-8429-6c024284bf8e
# ╠═c6bfd2a4-df9d-4f83-882d-d1bb1c1a3ecb
# ╠═09cce5a5-713d-445c-aadf-34f9a820be56
# ╠═ea8d3c39-e086-4db0-b50f-f1b64bc992cf
# ╠═d758e312-70d8-41e3-a1dc-cc6ce8a744db
# ╟─ee1b032c-e614-4251-90a7-4f50da3b61d1
# ╟─0763ed96-04bb-43b3-8c40-fa58344ea96f
# ╟─33ec1636-17f6-4e89-a42f-e368df94b965
# ╟─65562ef7-93eb-4d05-98fe-6689c26c4063
# ╟─f58afeb2-8704-4d1a-9f12-dd65ab05dc87
