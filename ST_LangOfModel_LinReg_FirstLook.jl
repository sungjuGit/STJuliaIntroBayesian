using Distributions
using StatsPlots
using StatsBase
using CSV
using DataFrames
using LinearAlgebra
using Random
using Turing


# helper function for getting probability/credibility interval
PI(d; p = 0.95) = [quantile(d, (1-p)/2), quantile(d, (1+p)/2)]

d = DataFrame(CSV.File("data/Howell1.csv"));
d2 = d[d.age .>= 18,:];  # select only those data from adults

scatter(d2.weight, d2.height)

p = hline([0, 272]; ylims=(-100, 400), xlabel="weight", ylabel="hegiht", legend=false)
title!("β ~ Normal(μ = 0, σ = 10)")

x̄ = mean(d2.weight)
xlims = extrema(d2.weight)  # getting min and max in one pass

# see what the normal distributions for α and β give
Random.seed!(2971)
N = 100
α = rand(Normal(178, 20), N);
β = rand(Normal(0, 10), N);
for (α_, β_) ∈ zip(α, β)
    plot!(x -> α_ + β_ * (x - x̄); xlims=xlims, c=:black, alpha=0.3, legend=false)
end
display(p)


β = rand(LogNormal(0, 1), 10_000)
density(β, xlims=(0, 5), bandwidth=0.1)

# see what the log-normal distribution for β gives
p = hline([0, 272]; ylims=(-100, 400), xlabel="weight", ylabel="hegiht")
title!("β ~ LogNormal(μ = 0, σ = 1)")
for (α_, β_) ∈ zip(α, β)
  plot!(x -> α_ + β_ * (x - x̄); xlims=xlims, c=:black, alpha=0.3, legend=false)
end
display(p)


# define our Bayesian statistics model for linear regression
@model function height_regr_model(weight, height)
    α ~ Normal(178, 20)
    β ~ LogNormal(0, 1)
    μ = α .+ β * (weight .- x̄)
    σ ~ Uniform(0, 50)
    height ~ MvNormal(μ, σ)
end

m = sample(height_regr_model(d2.weight, d2.height), NUTS(), 1000)

m_df_var = select(DataFrame(m), [:α, :β, :σ])
describe(m_df_var)

round.(cov(Matrix(m_df_var)), digits=3)


# now take 1000 samples from the posterior distribution obtained from the MC method
samples = sample(m, 1000)

# Use the mean values of α and β to plot the "best" fit line
α_MAP = mean(samples[:α])
β_MAP = mean(samples[:β])

p = @df d2 scatter(:weight, :height; alpha=0.3)
plot!(x -> α_MAP + β_MAP*(x-x̄))


# Let's reduce the sample size to 10 to explore posterior distributions of α and β
N = 10
dN = d2[1:N,:]

@model function height_regr_model_N(weight, height)
    α ~ Normal(178, 20)
    β ~ LogNormal(0, 1)
    x̄N = mean(weight)
    μ = α .+ β * (weight .- x̄N)
    σ ~ Uniform(0, 50)
    height ~ MvNormal(μ, σ)
end

mN = sample(height_regr_model_N(dN.weight, dN.height), NUTS(), 1000)

# sample 20 values of α and β from the posterior
samplesN = sample(mN, 20)
samplesN_df = DataFrame(samplesN);

xlims = extrema(d2.weight)
ylims = extrema(d2.height)
p = @df dN scatter(:weight, :height; xlims=xlims, ylims=ylims, label=false)
title!("N = $N"; xlab="weight", ylab="height")

x̄N = mean(dN.weight)
for (α_, β_) ∈ zip(samplesN_df.α, samplesN_df.β)
    plot!(x -> α_ + β_ * (x-x̄N); c="black", alpha=0.3, legend=false)
end
display(p)


# Let's now go back to our original posterior distribution obtained from all the adult data
samples = sample(m, 1000)
samples_df = DataFrame(samples)

# posterior distribution of μ for one particular value of weight
μ_at_50 = @. samples_df.α + samples_df.β * (50 - x̄);
density(μ_at_50; lw=2, xlab="μ|weight=50")


# now do the same for all weight values in the data
μ = map(w -> samples_df.α .+ samples_df.β * (w - x̄), d2.weight)
# 352-element vector where each element is a vector of 1000 elements
# these 1000 elements are obtained from 1000 combinations of α and β sampled from the posterior
μ = hcat(μ...) # turn the above array into 1000 x 352 matrix

# plot the first 100 values of μ
p = plot()
for i in 1:100
    scatter!(d2.weight, μ[i,:]; c=:blue, alpha=0.2, label=false)
end
display(p)


# Now, consider μ for a defined weight sequence (not the weight values from the data)
weight_seq = 25:70

μ_seq = map(w -> samples_df.α .+ samples_df.β * (w - x̄), weight_seq)
μ_seq = hcat(μ_seq...)

μ_seq_mean = mean.(eachcol(μ_seq))
μ_seq_PI = PI.(eachcol(μ_seq))
μ_seq_PI = vcat(μ_seq_PI'...) # convert vector of vector into 46x2 matrix # if using link


@df d2 scatter(:weight, :height; alpha=0.2, xlab="weight", ylab="height")
plot!(weight_seq, μ_seq_mean; c=:black)
plot!(weight_seq, [μ_seq_mean μ_seq_mean]; c=:black, fillrange=μ_seq_PI, fillalpha=0.3, label=false)


# We now do the similar simulation while including the effect of σ on the posterior distribution
sim_height = map(w -> rand.(Normal.(samples_df.α .+ samples_df.β * (w - x̄), samples_df.σ)), weight_seq)
sim_height = hcat(sim_height...);

height_PI = PI.(eachcol(sim_height))
height_PI = vcat(height_PI'...);

plot!(weight_seq, [μ_seq_mean μ_seq_mean]; c=:black, fillrange=height_PI, fillalpha=0.3)
