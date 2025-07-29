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
## Data fitting of optoLARG activation/deactivation dynamics after light stimulation
This notebook contains the code to fit an ODE model to the experimental data of optoLARG activition after local light stimulation as described in the paper by Heydasch et al. \


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
	signal_step(t) = (150 <= t <= 200)
	@register_symbolic signal_step(t) 

	# Use hyperbolic tangent to have continuous derivatives for data fitting as light signal function
	signal_tanh(t) = (tanh(100*(t-150))/2 - tanh(100*(t-200))/2)
	@register_symbolic signal_tanh(t)
	
end

# ╔═╡ 247533da-69ba-483c-8025-004c32a1bf9c
# Check the light signal functions 
begin
	x = LinRange(148, 203, 400)
	y1 = signal_step.(x)
	y2 = signal_tanh.(x)
	plot(x, [y1, y2], labels =["Step function" "Tanh"], lw = 2, title="Comparison of light signal functions")
end

# ╔═╡ d0339df7-b829-4d10-b0d6-817d037b0c58
md"""
##### - Import optoLARG data
"""

# ╔═╡ 27dff155-efb6-4574-b97a-63dd2a74a4f6
# Import data saved in txt files (Python preprocessing pickle file)
begin
	path_to_data = "../Data/"
	mean_optoLARG = readdlm(Base.Filesystem.joinpath(path_to_data, "mean_optolarg_dynamics.txt"), ' ', Float64, '\n')[:, 1]
	optolarg_timeframe = LinRange(0, 390, 40)
end

# ╔═╡ c0ea9b5d-658f-489c-bcdd-b7ce45e13232
# Visualise data
begin
	plot(optolarg_timeframe, mean_optoLARG, label = "optoLARG", fontfamily="Arial", color = "#800080", title= "optoLARG dynamics at focal adhesions after light pulse")
end

# ╔═╡ 487b4ff4-3e29-48d0-bb6d-9d9ec048de1c
md"""
### Data fitting
"""

# ╔═╡ bf9ab78e-2454-4570-98b9-e54249ead1e7
# Define simple linear model for optoLARG dynamics
@mtkmodel dynamics begin
	
	@parameters begin
		Kon
		Koff
	end
	
	@variables begin
		GEF(t), [bounds = (0, 10)]
		Signal(t)
	end
	
	@equations begin
		D(GEF) ~ Kon * Signal - Koff * GEF
		Signal ~ signal_tanh(t)
	end
end

# ╔═╡ a893dc0a-388a-4f72-880d-5eb67aeb6ce8
@mtkbuild model = dynamics()

# ╔═╡ d7de87d3-4f2e-4468-8a53-383085c0d291
begin

	# Define some initial parameters for the ODE model
	parameter_guess = [0.1, 0.2]

	# Build ODE problems with guess parameters and initial conditions
	prob = ODEProblem(model, 
		[model.GEF => 0], 
		(140.0, 390.0), parameter_guess, jac = true) 
end

# ╔═╡ 8bb63082-d079-40e1-a100-01144f08d060
# Define MSE loss function to estimate parameters
function MSE_loss(new_parameters)

	# Update the ODE problem with the new set of parameters
	new_prob = remake(prob, p=new_parameters)

	# Solve the new ODE problem (saveat = numerical estimate time interval)
	new_sol = solve(new_prob, AutoVern9(Rodas4P()), saveat = 10, maxiters = 1e7, dtmax = 0.05, verbose = false)
	new_sol_array = new_sol[model.GEF]

	# Determine the loss (sum of squares)
	if length(new_sol_array) == length(mean_optoLARG[15:end])
		loss = sum(abs2, new_sol_array .- mean_optoLARG[15:end])
	else
		loss = 1000
	end

	return loss
end

# ╔═╡ c61ef670-92e9-4169-bcfc-73901faf066f
begin
	# Define optimization function and differentiation method
	optimisation_f = Optimization.OptimizationFunction((x, p) -> MSE_loss(x), Optimization.AutoForwardDiff())

	# Keep track of loss values during iterations
	track_loss = []

	# Define upper and lower bound for parameters and initial conditions
	lower_b = repeat([0.0001], 2)
	upper_b = repeat([10], 2)
	
	# Define the optimisation problem
	optimisation_prob = Optimization.OptimizationProblem(optimisation_f, parameter_guess, lb = lower_b, ub = upper_b)

end

# ╔═╡ 01e276d1-1418-4fc2-b75a-edee7e5fad4f
# Define callback function that is run at each iteration
callback! = function(parameters, loss)

	# Add current loss value and parameter to tracking arrays
	push!(track_loss, loss)
	
	# Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
	return false
end

# ╔═╡ 66c89b70-dce7-4851-b962-202b3e60065e
begin 

	# Clear the tracking arrays if not empty
	if length(track_loss) > 0
		deleteat!(track_loss, 1:length(track_loss))
	end

	# Run the optimisation
	optim_results = Optimization.solve(optimisation_prob, LBFGS(), callback = callback!, maxiters = 100, progress = true)
end

# ╔═╡ 668d5e73-022a-4f45-8cd6-6218744f0c7f
begin

	# Solve the ODE problem with optimal parameters
	pred_sol = solve(remake(prob, p=optim_results, tspan=(140, 390)), saveat=1, dtmax=0.05)
	scatter(optolarg_timeframe[12:end], mean_optoLARG[12:end], label = "Data", lw=2, title = "Fitted optoLARG dynamics")
	plot!(pred_sol, label = "Prediction", lw=2, ylabel = "Fold-change \nnormalised to baseline", xlabel = "Time [s]")
end

# ╔═╡ 238ea5ce-3e2b-45c6-bb9d-abb82fca9a0d
begin
	# Save parameters
	saved_p = [0.0123462, 0.0700499]

	# Solve the ODE with the best parameters and plot
	saved_pred_sol = solve(remake(prob, p=saved_p, tspan=(120, 400)), saveat=1, dtmax=0.05) 
	scatter(optolarg_timeframe[12:end], mean_optoLARG[12:end], label = "Data", lw=2, title = "Fitted optoLARG dynamics")
	plot!(saved_pred_sol, label = "Prediction", lw=2, ylabel = "Fold-change \nnormalised to baseline", xlabel = "Time [s]")
	#savefig("../Plots/optolarg_plot.svg")
end

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

# ╔═╡ 34293c00-0b83-46ff-ad79-c8b44277671d
begin
	# Transform back ODE parameters to their symbolic expression
	symbolic_param = [
		model.Kon => saved_p[get_idxparam(model.Kon, prob)],
		model.Koff => saved_p[get_idxparam(model.Koff, prob)]
	]
end

# ╔═╡ Cell order:
# ╟─ffd55ae0-dbed-474b-9b89-10bcf3bdcfe8
# ╟─8277b2e6-6cf6-4f4a-ae4b-274548edb651
# ╟─a97176f9-7dd2-4870-9a80-b91e25ba8b97
# ╟─e44a2ff8-7326-4080-813e-ffbeb4c4b648
# ╟─52bdbc46-53fc-42ca-8fec-e43522c30063
# ╟─247533da-69ba-483c-8025-004c32a1bf9c
# ╟─d0339df7-b829-4d10-b0d6-817d037b0c58
# ╟─27dff155-efb6-4574-b97a-63dd2a74a4f6
# ╟─c0ea9b5d-658f-489c-bcdd-b7ce45e13232
# ╟─487b4ff4-3e29-48d0-bb6d-9d9ec048de1c
# ╟─bf9ab78e-2454-4570-98b9-e54249ead1e7
# ╟─a893dc0a-388a-4f72-880d-5eb67aeb6ce8
# ╟─d7de87d3-4f2e-4468-8a53-383085c0d291
# ╟─8bb63082-d079-40e1-a100-01144f08d060
# ╟─01e276d1-1418-4fc2-b75a-edee7e5fad4f
# ╟─c61ef670-92e9-4169-bcfc-73901faf066f
# ╟─66c89b70-dce7-4851-b962-202b3e60065e
# ╟─668d5e73-022a-4f45-8cd6-6218744f0c7f
# ╟─238ea5ce-3e2b-45c6-bb9d-abb82fca9a0d
# ╟─34293c00-0b83-46ff-ad79-c8b44277671d
# ╟─f58afeb2-8704-4d1a-9f12-dd65ab05dc87
