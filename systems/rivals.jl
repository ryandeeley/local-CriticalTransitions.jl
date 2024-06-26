"""
    rivals!(du, u, p, t)
In-place definition of the Rivals system.

See also [`rivals`](@ref).
"""
function rivals!(du, u, p, t)
    x, y = u
    ϵ, α₁, α₂, β₁, β₂ = p[1]

    du[1] = x*(x-α₁)*(1-x)-β₁*x*y
    du[2] = ϵ*(y*(y-α₂)*(1-y)-β₂*x*y)

end

"""
    rivals(u, p, t)
Out-of-place definition of the Rivals system.

See also [`rivals!`](@ref).
"""
function rivals(u, p, t)
    x, y = u    
    ϵ, α₁, α₂, β₁, β₂ = p[1]

    dx = x*(x-α₁)*(1-x)-β₁*x*y
    dy = ϵ*(y*(y-α₂)*(1-y)-β₂*x*y)

    SA[dx, dy]
end

"""
    rivals_ϵσ(ϵ,σ)
A shortcut command for returning a StochSystem of the Rivals system in a default setup with multiplicative isotropic noise. 
    
This setup fixes the parameters α₁ = 0.1, α₂ = 0.3, β₁ = 0.18, β₂ = 0.1 and leaves the value of the time-scale parameter ϵ as a function argument. The prescribed noise process is multiplicative and isotropic: the variables are peturbed by independently drawn identical Gaussian white noise realisations (with standard deviation σ - the other function argument) that are multiplied by the variables' current value.
"""
function rivals_ϵσ(ϵ, σ; save_everystep::Bool = false) # a convenient two-parameter version of the FitzHugh Nagumo system 
    # defining the StochSystem
    f(u,p,t) = rivals(u,p,t);
    α₁ = 0.1; α₂ = 0.3; β₁ = 0.18; β₂ = 0.1; # standard parameters without ϵ (time-scale separation parameter)
    pf_wo_ϵ = [α₁, α₂, β₁, β₂]; # parameter vector without ϵ
    u = zeros(2);
    g(u,p,t) = multiplicative_idx(u,p,t,[true,true]);
    pg = Float64[]; 
    Σ = [1 0; 0 1];
    process = WienerProcess(0.,[0.,0.]; save_everystep);
    StochSystem(f, vcat([ϵ], pf_wo_ϵ), u, σ, g, pg, Σ, process)
end;