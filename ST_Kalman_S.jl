module KalmanFilter

using LinearAlgebra

export Kalman, update, predict, K, next

const Matrix = AbstractArray{T, N} where {N, T <: Real}
const Vector = AbstractArray{T, N} where {N, T <: Real}

"""
Defines the Kalman Filter object.
"""
mutable struct Kalman
    A # State transition matrix
    Q # Process covariance matrix
    H # Measurement mapping
    R # Measurement covariance matrix
    m # Current estimate mean
    P # State covariance matrix.  Current estimate uncertianty
    
    function Kalman(A::Matrix, Q::Matrix, H::Matrix, R::Matrix, m::Vector, P::Matrix)
        new(A, Q, H, R, m, P)
    end

    # Creates a Kalman-Filter object for the scalar case.
    Kalman(A::Real, Q::Real, H::Real, R::Real, m::Real, P::Real) = new(A, Q, H, R, m, P)

end



#  K(Kalman): Computes the Kalman Gain based on a model.

function K(k::Kalman)
    K(k.P, k.H, k.R)
end

#  K(P, H, R): Computes the Kalman Gain based the matrices.
function K(P, H, R)
    P * transpose(H) * (H * P * transpose(H) + R)^-1
end



#   predict(k::Kalman): Predict next state based on the dynamic process model.
function predict(k::Kalman)
    k.m = k.A * k.m
    k.P = k.A * k.P * transpose(k.A) + k.Q
    (state=k.m, cov=k.P)
end


# update(k::Kalman, y): Compute the filtered distribution.
function update(k::Kalman, y)
    kalman_gain = K(k.P, k.H, k.R)
    k.m = k.m + kalman_gain * (y - k.H * k.m)
    k.P = (I - kalman_gain * k.H) * k.P * transpose(I - kalman_gain * k.H) + 
        kalman_gain * k.R * transpose(kalman_gain)
    (state=k.m, cov=k.P, gain=kalman_gain)
end


# next(k::Kalman, y): Compute a complete Kalman step. 
function next(k::Kalman, y)
    newInstance = deepcopy(k)
    p = predict(newInstance)
    up = update(newInstance, y)
    (filter = newInstance, updated = up.state, predicted = p.state, cov = up.cov, gain = up.gain)
end


end