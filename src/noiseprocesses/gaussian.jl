#include("../StochSystem.jl")
#include("../utils.jl")

using RandomNumbers.Xorshifts

# """
#     correlated_wienerprocess(sys::StochSystem)
# Returns a Wiener process with dimension `length(sys.u)` and covariance matrix `sys.Σ`.

# This function is based on the [`CorrelatedWienerProcess`](https://noise.sciml.ai/stable/noise_processes/#DiffEqNoiseProcess.CorrelatedWienerProcess) of [`DiffEqNoiseProcess.jl`](https://noise.sciml.ai/stable/), a component of `DifferentialEquations.jl`. The initial condition of the process is set to the zero vector at `t=0`.
# """
function correlated_wienerprocess(sys::StochSystem)
    # Returns a Wiener process for given covariance matrix and dimension of a StochSystem
    if is_iip(sys.f)
        W = CorrelatedWienerProcess!(sys.Σ, 0.0, zeros(length(sys.u)))
    else
        W = CorrelatedWienerProcess(sys.Σ, 0.0, zeros(length(sys.u)))
    end
    W.save_everystep = sys.process.save_everystep
    W
end;

## functions for creating a correlated Ornstein-Uhlenbeck process

function construct_correlatedOUprocess(dW, W, dt, u, p, t, rng, Σ::CovMatrix, ou::DiffEqNoiseProcess.OrnsteinUhlenbeck)
    X = ou;
    Γ = cholesky(Σ).L; # this represents the (lower-diagonal matrix) from the cholesky decomposition of the covariance matrixz
    if typeof(dW) <: AbstractArray
        rand_val = DiffEqNoiseProcess.wiener_randn(rng, dW)
    else
        rand_val = DiffEqNoiseProcess.wiener_randn(rng, typeof(dW))
    end
    Wtilde = inv(Γ)*W[end]; # the current value of the individual processes
    drift = X.μ .+ (Wtilde .- X.μ) .* exp.(-X.Θ * dt)
    diffusion = X.σ .* sqrt.((1 .- exp.(-2X.Θ * dt)) ./ (2X.Θ)) 
    # "drift .+ rand_val .* diffusion" represents the new value of the individual processes
    Γ*(drift .+ rand_val .* diffusion .- Wtilde) # the difference between the new and the old value of the entire process
end

function CorrelatedOrnsteinUhlenbeckProcess(Σ::CovMatrix, ou::DiffEqNoiseProcess.OrnsteinUhlenbeck, t0, W0, Z0 = nothing;
    rng = Xorshifts.Xoroshiro128Plus(rand(UInt64)))
    drift(dW, W, dt, u, p, t, rng) = construct_correlatedOUprocess(dW, W, dt, u, p, t, rng, Σ, ou);
    bridge(dW, W, dt, u, p, t, rng) = nothing; 
    NoiseProcess{false}(t0, W0, Z0, drift, bridge; rswm = RSWM(), rng)
end

# """
#     correlated_ornsteinuhlenbeckprocess(sys::StochSystem)
# Returns an Ornstein-Uhlenbeck process with dimension `length(sys.u)` and the covariance matrix `sys.Σ`.

# This function is a custom noise process from the [`DiffEqNoiseProcess.jl``](https://docs.sciml.ai/DiffEqNoiseProcess/stable/) package. The initial condition of the process at `t = 0` is as prescribed (by the user) in the setup.
# """
function correlated_ornsteinuhlenbeckprocess(sys::StochSystem)
    # Returns a Wiener process for given covariance matrix and dimension of a StochSystem
    if is_iip(sys.f)
        ErrorException("Out-of-place method not implemented yet.")
    else
        W = CorrelatedOrnsteinUhlenbeckProcess(sys.Σ, sys.process.dist, 0.0, sys.process.u[1])
    end
    W
end;