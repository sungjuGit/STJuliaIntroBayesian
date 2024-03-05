using DifferentialEquations, Plots
using CSV, DataFrames
using Turing

function lotka_volterra!(du, u, p, t)

  # Unpack the values so that they have clearer meaning
  x, y = u
  bx, mx, by, my = p

  # Define the ODE
  du[1] = (bx - y * mx) * x
  du[2] = (x*by - my) * y

end

# Model parameters
p = [1.1, 0.5, 0.1, 0.2]

# Initial conditions
u0 = [1, 1]

# Timespan of the solution
tspan = (0.0, 40.0)

prob = ODEProblem(lotka_volterra!, u0, tspan, p)

sol = solve(prob)

plot(sol)


# load data and determine parameters

data = CSV.read("./data/lv_pop_data.csv", DataFrame)
pop_data = Array(data)'

time_plot=0:2:30;
plot(time_plot, pop_data[1, :], label=false);
plot!(time_plot, pop_data[2, :], label=false);
scatter!(time_plot, pop_data[1, :], label="Prey");
scatter!(time_plot, pop_data[2, :], label="Pred")


@model function fitlv(data)

    σ ~ InverseGamma(2, 3)

    bx ~ truncated(Normal(1, 0.5), 0, 2)
    mx ~ truncated(Normal(1, 0.5), 0, 2)
    by ~ truncated(Normal(1, 0.5), 0, 2)
    my ~ truncated(Normal(1, 0.5), 0, 2)

    param = [bx, mx, by, my]
    prob = ODEProblem(lotka_volterra!, u0, (0.0, 30), param)
    predicted = solve(prob, Tsit5(), saveat=2)

    for i = 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ)
    end
end

model = fitlv(pop_data)

posterior = sample(model, NUTS(0.6), 10000) 

plot(posterior)

birth_prey = sample(posterior[:bx], 100)
mort_prey = sample(posterior[:mx], 100)
birth_pred = sample(posterior[:by], 100)
mort_pred = sample(posterior[:my], 100)

solutions = []

for i in 1:length(birth_prey)

    p = [birth_prey[i], mort_prey[i], birth_pred[i], mort_pred[i]];
    problem = ODEProblem(lotka_volterra!, u0, (0.0, 30.0), p);
    push!(solutions, solve(problem, saveat = 0.1));

end
    
p_mean = [mean(birth_prey), mean(mort_prey), mean(birth_pred), mean(mort_pred)];

problem_mean = ODEProblem(lotka_volterra!, u0, (0.0,30.0), p_mean);
push!(solutions, solve(problem_mean, saveat = 0.1));


plot(solutions[1], alpha=0.2, color="blue");

for i in 2:(length(solutions) - 1)
    plot!(solutions[i], alpha=0.2, legend=false, color="blue");
end

plot!(solutions[end], lw = 2, color="red");
      
# Comparing inference with the data
scatter!(time_plot, pop_data[1, :], color = "blue");
scatter!(time_plot, pop_data[2, :], color = "orange")
  

