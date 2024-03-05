include("./ST_Kalman_S.jl")
using .KalmanFilter
using LinearAlgebra


# example for 2nd order linear Kalman filter
# x = [x1 x2 x1dot x2dot x1dotdot x2dotdot]
# measurements are done for x1, x2, x1dotdot, and x2dotdot

x_0 = zeros(6, 1)

Σ_0 = Matrix{Float64}(I, 6, 6) .* 1000

Δ_t = 0.1

A = [1 0 Δ_t 0 0.5*Δ_t^2 0;
    0 1 0 Δ_t 0 0.5*Δ_t^2;
    0 0 1 0 Δ_t 0;
    0 0 0 1 0 Δ_t;
    0 0 0 0 1 0;
    0 0 0 0 0 1]

H = [1 0 0 0 0 0;
    0 1 0 0 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1]

R = [2 0 0 0;
    0 10 0 0;
    0 0 0.4 0;
    0 0 0 0.4]

Q = Matrix{Float64}(I, 6, 6) * 0.0001



filter = KalmanFilter.Kalman(A, Q, H, R, x_0, Σ_0)

n = 2  # number of measurement data

predicted = zeros(Float64, 6, n)
updated = zeros(Float64, 6, n)
covariances = zeros(Float64, 6, 6, n)
gains = zeros(Float64, 6, 4, n)

for i in 1:n # = eachrow(data)    

    y_cur = [1 1 1 1]' # should be replaced with the measurement data in each time step

    filter_next = KalmanFilter.next(filter, y_cur)
    
    predicted[:, i] = filter_next.predicted
    updated[:, i] = filter_next.updated
    covariances[:,:,i] = filter_next.cov
    gains[:,:,i] = filter_next.gain
    
    filter = filter_next.filter  # use recursion

end

