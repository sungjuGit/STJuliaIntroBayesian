using CairoMakie, Turing, DataFrames, CSV, Statistics, StatsBase, ColorSchemes

update_theme!(fontsize=18, resolution=(0.9*500, 0.9*380))

# F. Waqar, S. Patel, C. Simon. "A tutorial on the Bayesian statistical approach to inverse problems" APL Machine Learning.

# forward model for lime temperature
function θ(t, λ, θ₀, θᵃⁱʳ)
    if t < 0.0
        return θ₀
    end
    return θᵃⁱʳ + (θ₀ - θᵃⁱʳ) * exp(-t / λ)
end

#probabilistic model of the temperature measurements
#we use the forward model to construct a probabalistic model of the measured lime temperature. 
#we assume any observed measurement [°C] of the lime temperature at time 
#is a realization of a random variable 
#, a Gaussian distribution
# \begin{equation}
# \Theta_{\text{obs}} \mid \lambda, \theta_0, \theta^{\text{air}}, \sigma \sim \mathcal{N}(\theta(t; \lambda, \theta_0, \theta^{\text{air}}), \sigma^2)
# \end{equation}

# the variance in the measurement originates from measurement noise and zero-mean residual variability.
# we treat multiple measurements as independent and identically distributed. 

#parameter inference
# task: infer the parameter in the model of the lime temperature.
# sub-tasks: infer (also treated as random variables):
#   variance of the measurement noise, 
#   initial lime temperature, even though we will take a (noisy) measurement of it.
#   air temperature, even though we will take a (noisy) measurement of it.

θ₀_obs = 6.54 # °C
θᵃⁱʳ_obs = 18.47 # °C

data = CSV.read("./data/lime_temp_param_id.csv", DataFrame)

begin
	local fig = Figure()
	local ax  = Axis(fig[1, 1], xlabel="time, t [hr]", 
		ylabel="lime temperature, θ [°C]")
	scatter!(data[:, "t [hr]"], data[:, "θ_obs [°C]"], color=Cycled(1),
		label=rich("{(t", subscript("i"), ", θ", subscript("i, obs"), ")}"))
	scatter!([0], [θ₀_obs], color=Cycled(1))
	hlines!([θᵃⁱʳ_obs], color=Cycled(3), linestyle=:dash, 
		label=rich("θ", superscript("air"), subscript("obs")))
	axislegend(position=:rb)
	fig
end


# implementation in Turing.jl, for specifiying prior and forward model
@model function measure_lime_temp_time_series(data)
    # prior distributions
	λ    ~ truncated(Normal(1.0, 0.3), 0.0, nothing) # hr
    σ    ~ Uniform(0.0, 1.0) # °C
    θ₀   ~ Normal(θ₀_obs, σ) # °C
    θᵃⁱʳ ~ Normal(θᵃⁱʳ_obs, σ) # °C

    # probabilistic forward model
    for i = 1:nrow(data)
		# the time stamp
        tᵢ = data[i, "t [hr]"]
		# the model prediction
        θ̄ = θ(tᵢ, λ, θ₀, θᵃⁱʳ)
		# the probabilistic forward model
        data[i, "θ_obs [°C]"] ~ Normal(θ̄, σ)
	end
end


begin
	mlts_model = measure_lime_temp_time_series(data)
		
	nb_samples = 2_500 # per chain
	nb_chains = 4      # independent chains
	chain = DataFrame(
		sample(mlts_model, NUTS(), MCMCSerial(), nb_samples, nb_chains)
	)
end


#we compare the dist'n of lambda
#over the nb_chains=4 independent chains (a convergence diagnostic–-they should approximately match).
begin
	local fig = Figure()
	local ax = Axis(fig[1, 1], xlabel="λ [hr]", ylabel="# samples")
	for (i, c) in enumerate(groupby(chain, "chain"))
		hist!(c[:, "λ"], color=(ColorSchemes.Accent_4[i], 0.5))
	end
	fig
end

μ_λ = mean(chain[:, "λ"]) # hr
σ_λ = std(chain[:, "λ"]) # hr
ci_λ = [percentile(chain[:, "λ"], 5.0), percentile(chain[:, "λ"], 95.0)]


begin
	local fig = Figure()
	local ax  = Axis(fig[1, 1], xlabel="λ [hr]", ylabel="# samples", 
		title="posterior dist'n of Λ")
	hist!(chain[:, :λ])
	ylims!(0, nothing)
	vlines!([μ_λ], linestyle=:dash, color=Cycled(2))
	lines!(ci_λ, zeros(2), color="black", linewidth=10)
	fig
end



begin
	local fig = Figure()
	local ax  = Axis(fig[1, 1], xlabel="time, t [hr]", 
		ylabel="lime temperature, θ [°C]")

	n_trajectories = 500
	t = range(-0.2, 10.0, length=100)
	for row in eachrow(chain[1:1+n_trajectories,:])
		lines!(ax, t, θ.(t, row[:λ], row[:θ₀], row[:θᵃⁱʳ]), 
			color=("orange", 0.05))
	end
	scatter!(data[:, "t [hr]"], data[:, "θ_obs [°C]"], color=Cycled(1),
		label=rich("{(t", subscript("i"), ", θ", subscript("i, obs"), ")}"))
	scatter!([0], [θ₀_obs], color=Cycled(1))
	hlines!([θᵃⁱʳ_obs], color=Cycled(3), linestyle=:dash, 
		label=rich("θ", superscript("air"), subscript("obs")))
	axislegend(position=:rb)

	xlims!(-0.2, 10)
	fig
end

# finally, we compute the mean and variance of the posterior for 
#, which we will used in the time reversal problem we tackle next.

μ_σ = mean(chain[:, "σ"])
σ_σ = std(chain[:, "σ"])


# time reversal
t′ = 0.68261 # hr

#the heat transfer experiment
#we conduct a second lime heat transfer experiment.

θᵃⁱʳ_obs_2 = 18.64 # °C

# at time t', we measure the lime temperature 

θ′_obs = 12.16 # °C

data2 = DataFrame("t [hr]"=>[t′], "θ_obs [°C]"=>[θ′_obs]) # put into a data frame for convenience

@model function measure_lime_temp_later(data2)
    # prior distributions
	λ    ~ truncated(Normal(μ_λ, σ_λ), 0.0, nothing) # hr
    σ    ~ truncated(Normal(μ_σ, σ_σ), 0.0, nothing) # °C
    θ₀   ~ Uniform(0.0, 20.0) # °C
    θᵃⁱʳ ~ Normal(θᵃⁱʳ_obs_2, σ) # °C

    # probabilistic forward model
	t′ = data2[1, "t [hr]"]
	# the model prediction
	θ̄ = θ(t′, λ, θ₀, θᵃⁱʳ)
	# the likelihood
	data2[1, "θ_obs [°C]"] ~ Normal(θ̄, σ)
end

chain2 = DataFrame(
	sample(measure_lime_temp_later(data2), NUTS(), MCMCSerial(), nb_samples, nb_chains)
)

μ_θ₀ = mean(chain2[:, "θ₀"])
ci_θ₀ = [percentile(chain2[:, "θ₀"], 5.0), percentile(chain2[:, "θ₀"], 95.0)]

begin
	local fig = Figure()
	local ax  = Axis(fig[1, 1], xlabel="θ₀ [°C]", ylabel="# samples", 
		title="posterior dist'n of Θ₀")
	hist!(chain2[:, "θ₀"])
	ylims!(0, nothing)
	vlines!([μ_θ₀], linestyle=:dash, color=Cycled(2))
	lines!(ci_θ₀, zeros(2), color="black", linewidth=10)
	fig
end




