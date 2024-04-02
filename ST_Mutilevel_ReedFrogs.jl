using Turing
using DataFrames
using CSV
using Distributions
using StatsFuns
using StatsPlots
using StatsBase
using Random

default(label=false);

d = DataFrame(CSV.File("data/reedfrogs.csv"))
describe(d)

d.tank = 1:nrow(d)
d

# conventional single-level model
@model function frog_single_level(S, N, tank)

    a ~ filldist(Normal(0, 1.5), length(tank))  # offsets are defined for each tank
    p = logistic.(a)  # probability of survival or proportional survival
    S .~ Binomial.(N, p)

end

Random.seed!(1)
frog_single_level_ch = sample(frog_single_level(d.surv, d.density, d.tank), NUTS(200, 0.65, init_ϵ=0.5), 1000)
frog_single_level_df = DataFrame(frog_single_level_ch);

# multilevel model
@model function frog_multi_level(S, N, tank)

    σ ~ Exponential()
    ā ~ Normal(0, 1.5)

    a ~ filldist(Normal(ā, σ), length(tank))
    p = logistic.(a)
    S .~ Binomial.(N, p)

end

Random.seed!(1)
frog_multi_level_ch = sample(frog_multi_level(d.surv, d.density, d.tank), NUTS(200, 0.65, init_ϵ=0.2), 1000)
frog_multi_level_df = DataFrame(frog_multi_level_ch);


link_fun = (chain_df, dr) -> begin
    a = chain_df[:,"a[$(dr.tank)]"]
    p = logistic.(a)
    binomlogpdf.(dr.density, p, dr.surv)
end

single_level_surival = map( dr -> link_fun(frog_single_level_df, dr), eachrow(d) )
single_level_surival = hcat(single_level_surival...)

multi_level_survival = map( dr -> link_fun(frog_multi_level_df, dr), eachrow(d) )
multi_level_survival = hcat(multi_level_survival...);



# sample 10_000 samples again

post = sample(frog_multi_level_ch, 10000)
post_df = DataFrame(post)

propsurv_est = [
    logistic(mean(post_df[:,"a[$i]"]))
    for i ∈ 1:nrow(d)
]

scatter(propsurv_est, mc=:white, label="model", legend=:topright, xlab="tank", ylab="proportion survival", ylim=(-0.05, 1.05))
scatter!(d.propsurv, mc=:blue, ms=3, label="data")
hline!([mean(logistic.(post_df.ā))], ls=:dash, c=:black)
vline!([16.5, 32.5], c=:black)
annotate!([
        (8, 0, ("small tanks", 10)),
        (16+8, 0, ("medium tanks", 10)),
        (32+8, 0, ("large tanks", 10))
])



p1 = plot(xlim=(-3, 4), xlab="Log-odds survival", ylab="Density");
for r ∈ first(eachrow(post_df), 100)
    plot!(Normal(r.ā, r.σ), c=:black, alpha=0.2)
end
p1


sim_tanks = @. rand(Normal(post_df.ā[1:8000], post_df.σ[1:8000]));
p2 = plot(xlab="Probability survival", ylab="Density", xlim=(-0.1, 1.1));
density!(logistic.(sim_tanks), lw=2)

plot(p1, p2, size=(800, 400))





## Varying effects and the underfitting/overfitting trade-off
# Generate a mock data to test the models

ā = 1.5
σ = 1.5
nponds = 60
Ni = repeat([3, 10, 25, 35], inner=15);

a_pond = rand(Normal(ā, σ), nponds);    # mock "true" data

dsim = DataFrame(pond=1:nponds, Ni=Ni, true_a=a_pond);
dsim.p_true = logistic.(dsim.true_a);

Random.seed!(1)
dsim.Si = @. rand(Binomial(dsim.Ni, dsim.p_true));


# no pooling (treat each pond separately)
dsim.p_nopool = dsim.Si ./ dsim.Ni;

# partial pooling using the multi-level model
@model function pond_multi_level(Si, Ni) #, pond)

    σ ~ Exponential()
    ā ~ Normal(0, 1.5)
    a_pond ~ filldist(Normal(ā, σ), length(Ni))
    p = logistic.(a_pond)
    @. Si ~ Binomial(Ni, p)

end

Random.seed!(1)
pond_multi_level_ch = sample(pond_multi_level(dsim.Si, dsim.Ni), NUTS(), 1000)
pond_multi_level_df = DataFrame(pond_multi_level_ch);



dsim.p_partpool = [
    mean(logistic.(pond_multi_level_df[:,"a_pond[$i]"]))
    for i ∈ 1:nponds
];



nopool_error = @. abs(dsim.p_nopool - dsim.p_true)
partpool_error = @. abs(dsim.p_partpool - dsim.p_true);

plt = scatter(nopool_error, xlab="pond", ylab="absolute error", label = "no pooling")
scatter!(partpool_error, mc=:white, label = "partial pooling")

dsim.nopool_error = nopool_error
dsim.partpool_error = partpool_error

# group results according to the number of tadpoles
gb = groupby(dsim, :Ni)
pools = combine(gb, :nopool_error => mean, :partpool_error => mean, :pond => minimum, :pond => maximum)

nopool_mean = pools.nopool_error_mean
partpool_mean = pools.partpool_error_mean

pond_min = pools.pond_minimum # ranges of pond numbers with the same numbers of tadpoles
pond_max = pools.pond_maximum


for i in 1:length(pond_min)
    plot!([pond_min[i],pond_max[i]],[nopool_mean[i],nopool_mean[i]])
end

for i in 1:length(pond_min)
    plot!([pond_min[i],pond_max[i]],[partpool_mean[i],partpool_mean[i]], line=:dash)
end

plt