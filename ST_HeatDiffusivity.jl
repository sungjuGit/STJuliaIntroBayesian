using Turing, StatsPlots, DifferentialEquations, LinearAlgebra

# Define the parameters
true_D = 0.25

# Temporal domain
n_time_points = 6
tspan = (0, 0.3)
t_domain = range(tspan[1], stop=tspan[2], length=n_time_points)

# Spatial domain
n_space_points = 21
x_domain = range(0, stop=1, length=n_space_points)
dx = x_domain[2] - x_domain[1]

# Define the initial condition
u0 = exp.(-20 *(x_domain .- 0.5).^2);

# Define the discretized DE using finite difference method
function discretized_diffusion!(du, u, p, t)
    D = p[1]
    D2 = D/dx^2
    for i in 2:(n_space_points - 1)
        du[i] = D2 * (u[i+1] - 2 * u[i] + u[i-1])
    end
    du[1] = du[n_space_points] = 0 # Zero-flux boundary conditions
end

# Solve the discretized PDE to obtain true states
p_true = [true_D]
true_problem = ODEProblem(discretized_diffusion!, u0, tspan, p_true)
true_solution = solve(true_problem, Tsit5(), saveat=t_domain)

# Generate noisy observations
noise_std = 0.1
noisy_data = true_solution .+ randn(n_space_points,n_time_points) * noise_std;

# Plot the true solution and noisy observations
p = plot(x_domain, true_solution[:, 1], label="True Solution", lw=2, color=:blue, alpha=0.5)
scatter!(x_domain, noisy_data[:, 1], label="Noisy Observations", color=:red, legend=:topleft, markersize=3)
for i in 2:n_time_points
    plot!(x_domain, true_solution[:, i], lw=2, label="", color=:blue, alpha=0.5)
    scatter!(x_domain, noisy_data[:, i], color=:red, label="", markersize=3, markerstrokewidth=0)
end 
p



@model function fit_diffusion(observed_data, t_domain)

    # Define the prior distribution for the diffusion coefficient
    D ~ Beta(2, 3)
    # And the prior distribution for the noise
    σ ~ InverseGamma(10, 3)
    
    # Solve the discretized PDE with the current parameter value
    param = [ D ]
    problem = ODEProblem(discretized_diffusion!, u0, tspan, param)
    prediction = solve(problem, Tsit5(), saveat=t_domain)

    # Define the likelihood for the observed data    #observed_data ~ arraydist(Normal.(prediction, σ))
    for i in 1:length(prediction)
        observed_data[:, i] ~ MvNormal(prediction[i], σ^2 * I)
    end
    
end

# Run the sampler
model = fit_diffusion(noisy_data, t_domain)
chain = sample(model, NUTS(), 1000)
plot(chain)

# what if we observe only the first time step
model_short = fit_diffusion(noisy_data[:,1:2], t_domain[1:2])
chain_short = sample(model_short, NUTS(), 1000)
plot(chain_short)