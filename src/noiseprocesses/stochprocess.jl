#include("../StochSystem.jl")
#include("../utils.jl")
#include("gaussian.jl")

# """
#     stochprocess(sys::StochSystem)

# Translates the stochastic process specified in `sys` into the language required by the
# `SDEProblem` of `DynamicalSystems.jl`.
# """
# function stochprocess(sys::StochSystem)
#     if sys.process == "WhiteGauss"
#         if sys.Σ == I(length(sys.u))
#             return nothing
#         else
#             return gauss(sys)
#         end
#     else
#         ArgumentError("Noise process not yet implemented.")
#     end
# end

# sys.process == WhiteGauss()
# if sys.process == OrnsteinUhlenbeckProcess(args..., kwargs...)
#     return correlatedcorrelated(OrnsteinUhlenbeckProcess(args...,kwargs))
# end

# if sys.Σ = I(length(sys.u))
#     return sys.process
# elseif sys.process == WhiteGauss()
#     ## specification for correlated GWN
# elseif sys.process = OrnsteinUhlenbeckProcess()
#     ## specification for correlatedcorrelated
# else 
#     println("DIY")
# end

include("gaussian.jl")

"""
    stochprocess(sys::StochSystem)

Translates the stochastic process specified in `sys` into the language required by the
`SDEProblem` of `DynamicalSystems.jl`.
"""
function stochprocess(sys::StochSystem)
    if sys.Σ == I(length(sys.process.u[1])) # i.e. we can use the noise process sys.process directly
        return sys.process
    else
        if sys.process.dist == DiffEqNoiseProcess.WHITE_NOISE_DIST # i.e. we have a Gaussian White Noise process
            return correlated_wienerprocess(sys); # returns an oop (ip) Correlated Wiener Process in accordance with sys.Σ
        elseif fieldnames(typeof(sys.process.dist)) == (:Θ, :μ, :σ) # i.e. we have a Ornstein-Uhlenbeck process  
            return correlated_ornsteinuhlenbeckprocess(sys); # returns an oop (ip) Correlated Ornstein-Uhlenbeck Process in accordance with sys.Σ
        else
            ArgumentError("Noise process not yet implemented")
        end
    end
end

# function stochprocess(sys::StochSystem)
#     if sys.process[1] == "WhiteGauss"
#         if sys.Σ == I(length(sys.u))
#             return nothing
#         else
#             return gauss(sys) 
#         end
#     elseif sys.process[1] == "OrnsteinUhlenbeck"
#         γ, μ, σₙ, t0, N0 = sys.process[2]  
#         return OrnsteinUhlenbeckProcess(γ, μ, σₙ, t0, N0)
#     elseif sys.process[1] == "Custom"
#         return sys.process[2]
#     else
#         ArgumentError("Noise process not yet implemented.")
#     end
# end