using Random
using StatsBase
using Distributions
using StatsPlots
using StatsFuns

using Turing
using CSV
using DataFrames
using Optim


## 9.1 Good King Markov and His Island Kingdom

Random.seed!(1)
num_weeks = 10^5
positions = []
current = 10

for i ∈ 1:num_weeks
    # record current position
    push!(positions, current)
    # flip coin to generate proposal
    proposal = current + sample([-1, 1])
    # handle loops around
    proposal < 1 && (proposal = 10)
    proposal > 10 && (proposal = 1)
    # move?
    prob_move = proposal / current
    if (rand() < prob_move) current = proposal end
end

scatter(positions[1:100], xlab="week", ylab="island")

histogram(positions, xlab="island", ylab="number of weeks")



## 9.2 Metropolis algorithms

D = 5
T = 1000
Y = rand(MvNormal(zeros(D), ones(D)), T)
Rd = sqrt.(sum.(eachcol(Y.^2)))
density(Rd)
# As the dimension (D) increases, we end up sampling further and further away from the peak in the distribution (Y=0)



## 9.3 Hamiltonian Monte Carlo

# Let's use a simple case of a bivariate normal distribution

Random.seed!(7)

x = rand(Normal(), 50)
y = rand(Normal(), 50)
x = standardize(ZScoreTransform, x)
y = standardize(ZScoreTransform, y);


function U(q::Vector{Float64}; a=0, b=0.5, k=0, d=0.5)::Float64
    μx, μy = q
    U = sum(normlogpdf.(μx, 1, x)) + sum(normlogpdf.(μy, 1, y)) 
    U += normlogpdf(a, b, μx) + normlogpdf(k, d, μy)
    -U
end

function ∇U(q::Vector{Float64}; a=0, b=0.5, k=0, d=0.5)::Vector{Float64}
    μx, μy = q
    G₁ = sum(x .- μx) / 1^2 + (a - μx) / b^2  # ∂U/∂μx
    G₂ = sum(y .- μy) / 1^2 + (k - μy) / d^2  # ∂U/∂μy
    [-G₁, -G₂]
end


function HMC2(U, ∇U, ϵ::Float64, L::Int, current_q::Vector{Float64})

    q = current_q
    p = rand(Normal(), length(q))  # random flick - p is momentum
    current_p = p
    
    # make a half step for momentum at the beginning
    p -= ϵ * ∇U(q) / 2
    
    # initialize bookkeeping - saves trajectory
    qtraj = [q]
    ptraj = [p]
    
    # Alternate full steps for position and momentum
    for i ∈ 1:L
        q = q .+ ϵ * p  # full step for the position
        # make a full step for the momentum except at the end of trajectory
        if i != L
            p -= ϵ * ∇U(q)
            push!(ptraj, p)
        end
        push!(qtraj, q)
    end
    
    # Make a half step for momentum at the end
    p -= ϵ * ∇U(q) / 2
    push!(ptraj, p)
    
    # negate momentum at the end of trajectory to make the proposal symmetric
    p = -p
    
    # evaluate potential and kinetic energies at the start and the end of trajectory
    current_U = U(current_q)
    current_K = sum(current_p.^2)/2
    proposed_U = U(q)
    proposed_K = sum(p.^2)/2
    
    # accept or reject the state at the end of trajectory 
    # based in part on whether the total energy conservation is valid,
    # returning either the position at the end of the trajectory or the initial position
    accept = (rand() < exp(current_U + current_K - proposed_U - proposed_K))

    if accept
        current_q = q
    end
    
    (q=current_q, traj=qtraj, ptraj=ptraj, accept=accept)

end


Random.seed!(1)
Q = (q=[-0.1, 0.2],)  # initialize Q.q.  Note that Q is an array of named tuples generated from HMC2

step = 0.03
L = 11

n_samples = 4

plot_mu = scatter([Q.q[1]], [Q.q[2]], xlab="μx", ylab="μy", label = false)

for i ∈ 1:n_samples
    Q = HMC2(U, ∇U, step, L, Q.q)
    if n_samples < 10 
        cx, cy = [], []
        for j ∈ 1:L
            K0 = sum(Q.ptraj[j].^2)/2  # kinetic energy
            plot!(
                [Q.traj[j][1], Q.traj[j+1][1]],
                [Q.traj[j][2], Q.traj[j+1][2]],
                lw=1+2*K0,  # line width is proportional to kinetic energy
                c=:black,
                alpha=0.5,
                label=false
            )
            push!(cx, Q.traj[j+1][1])
            push!(cy, Q.traj[j+1][2])
        end
        scatter!(cx, cy, c=:white, ms=3, label = false)
    end
    scatter!([Q.q[1]], [Q.q[2]], shape=(Q.accept ? :circle : :rect), c=:blue, label = false)
end
plot_mu