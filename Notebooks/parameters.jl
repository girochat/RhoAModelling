using ModelingToolkit, DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity, ForwardDiff, Plots, SymbolicIndexingInterface

@variables t
    D = Differential(t)

# Define model 
@mtkmodel protein_dynamics begin

    @parameters begin
        x1_act
        x1_inact
        x2_act
        x2_inact
    end
    
    @variables begin
        x1(t)
        x2(t)
    end
    
    @equations begin
        D(x1) ~ x1_act - x1_inact * x1
        D(x2) ~ x2_act - x2_inact * x2
    end
end

@mtkbuild protein_model = protein_dynamics()

println(parameter_index(protein_model, protein_model.x1_act))



# Simulate data for true parameters
begin
    true_param =  [protein_model.x1_act => 0.02, 
        protein_model.x1_inact => 0.2, 
        protein_model.x2_act => 0.04, 
        protein_model.x2_inact => 0.4]
    simul_data_prob = ODEProblem(protein_model, 
        [protein_model.x1 => 0.5, 
        protein_model.x2 => 1], (0.0, 200.0), true_param, jac = true)

    simul_data_sol = solve(simul_data_prob, saveat=1, dtmax=0.5)

    # Import the data as array
    simul_data_x1 = simul_data_sol[protein_model.x1]
    simul_data_x2 = simul_data_sol[protein_model.x2]
end

println(simul_data_prob.p)

function loss(new_parameters)
	
	# Update the ODE problem with the new set of parameters
	new_ODE_prob = remake(simul_data_prob, p=new_parameters)

	# Solve the new ODE problem 
	new_sol = solve(new_ODE_prob, saveat = 1, dtmax=0.5)
	new_sol_array_x1 = new_sol[protein_model.x1]
	new_sol_array_x2 =  new_sol[protein_model.x2]

	# Determine the loss by computing the sum of squares
	if length(simul_data_x1) == length(new_sol_array_x1)
		loss = sum(abs2, new_sol_array_x1 .- simul_data_x1) + 
		sum(abs2, new_sol_array_x2 .- simul_data_x2)
	else
		loss = 1000
	end

	return loss, new_sol
end

# Define optimization function and differentiation method
optim_f = Optimization.OptimizationFunction((x, p) -> loss(x), 
	Optimization.AutoForwardDiff())
	
# Initial parameter guess
param_guess = [0.1, 0.1, 0.1, 0.1]
	
optim_prob = Optimization.OptimizationProblem(optim_f, param_guess)


# Run the optimisation
results = Optimization.solve(optim_prob, PolyOpt(), maxiters = 100)

#println(results.ps[protein_model.x1_act])
#sol = remake(simul_data_prob, p=results.u)
#println(sol.ps[protein_model.x1_act])
#println(simul_data_sol.ps[protein_model.x1_act])
#sol = solve(remake(simul_data_prob, p=results.u), saveat=0.5, dtmax = 0.5)
#println(sol.ps[protein_model.x1_act])
println(results.u)
