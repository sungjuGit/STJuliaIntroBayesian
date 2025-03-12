
## See the discussion in
# https://discourse.julialang.org/t/waic-computation/119553/2


function my_waic(chains, X, Y)
    dfc = @chain DataFrame(chains) begin
        select([:iteration, :chain, :σ, :β₀, :β₁])
    end

    n = length(Y)
    lppds = Vector{Float64}(undef, n)
    pwaics = Vector{Float64}(undef, n)

    for (i, (x, y)) in zip(X, Y) |> enumerate
        mus = dfc[:, :β₀] + dfc[:, :β₁] .* x
        log_pds = map((m, s) -> logpdf(Normal(m, s), y), mus, dfc[:, :σ])

        # Calculate lppd
        lppds[i] = logsumexp(log_pds) - log(length(log_pds))
        
        # Calculate penalty term (effective number of parameters)
        pwaics[i] = var(log_pds)
    end
    return lppds, pwaics
end

#### And then

lppds, pwaics = my_waic(chains, train_X, train_Y)
elpd = sum(lppds) - sum(pwaics)
WAIC = -2 * elpd
WAIC_SE = sqrt(length(lppds) * var(-2 .* (lppds - pwaics)))


### using ArviZ

ll = let
    plls = pointwise_loglikelihoods(model, chains)
    # Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
    log_likelihood_y = getindex.(Ref(plls), string.(keys(posterior_predictive)))
    (; Y=cat(log_likelihood_y...; dims=3))
end

idata = ArviZ.from_mcmcchains(chains;
    log_likelihood = ll,
    library = "Turing",
)

waic(idata)