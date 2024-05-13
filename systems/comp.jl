"""
Dynamical systems specification file
"""

# model of competing species

function comp!(du, u, p, t)
    x, y = u
    α₁,α₂,β₁,β₂,ε = p[1]

    du[1] = (x*(x-α₁)*(1-x)-β₁*x*y)/ε
    du[2] = y*(y-α₂)*(1-y)-β₂*x*y
end

function comp(u,p,t)
    x, y = u
    α₁,α₂,β₁,β₂,ε = p[1]

    dx = (x*(x-α₁)*(1-x)-β₁*x*y)/ε
    dy = y*(y-α₂)*(1-y)-β₂*x*y

    SA[dx, dy]
end

function comp_εσ(ε, σ; save_everystep::Bool = false, linear_diffusion::Bool = false) # a convenient two-parameter version of the COMP system 
    f(u,p,t) = comp(u,p,t);
    α₁ = 0.1; α₂ = 0.3; β₁ = 0.18; β₂ = 0.1; # standard parameters without ε (time-scale separation parameter)
    pf = vcat([α₁,α₂,β₁,β₂],[ε]); # parameter vector
    u = zeros(2);
    g(u,p,t) = linear_diffusion ? multiplicative_idx(u,p,t,[true,true]) : [√u[1] 0; 0 √u[2]]; 
    pg = Float64[]; 
    Σ = I(2);
    process = WienerProcess(0.,u; save_everystep);
    StochSystem(f, pf, u, σ, g, pg, Σ, process)
end;

# the log(⋅) transformed version of the comp system

function logcomp(u,p,t)
    u, v = u
    α₁,α₂,β₁,β₂,ε,σ = p[1]

    du = ((exp(u)-α₁)*(1-exp(u))-β₁*exp(v))/ε-σ^2/(2*exp(u))
    dv = (exp(v)-α₂)*(1-exp(v))-β₂*exp(v)-σ^2/(2*exp(v))

    SA[du, dv]
end

function logcomp_εσ(ε, σ; save_everystep::Bool = false, linear_diffusion::Bool = false) # a convenient two-parameter version of the COMP system 
    f(u,p,t) = logcomp(u,p,t);
    α₁ = 0.1; α₂ = 0.3; β₁ = 0.18; β₂ = 0.1; # standard parameters without ε (time-scale separation parameter)
    pf = vcat([α₁,α₂,β₁,β₂],[ε,σ]); # parameter vector
    u = zeros(2);
    g(u,p,t) = linear_diffusion ? idfunc : [exp(-u[1]/2); exp(-u[2]/2)]; 
    pg = [0.]; 
    Σ = I(2);
    process = WienerProcess(0.,u; save_everystep);
    StochSystem(f, pf, u, σ, g, pg, Σ, process)
end;

# the 2*sqrt(⋅) transformed version of the comp system

function twosqrtcomp(u,p,t)
    u, v = u
    α₁,α₂,β₁,β₂,ε,σ = p[1]

    du = ((u/2)*(u^2/4-α₁)*(1-u^2/4)-β₁*u*v^2/8)/ε-σ^2/(2*u)
    dv = (v/2)*(v^2/4-α₂)*(1-v^2/4)-β₂*u^2*v/8-σ^2/(2*v)

    SA[du, dv]
end

function twosqrtcomp_εσ(ε, σ; save_everystep::Bool = false, linear_diffusion::Bool = false) # a convenient two-parameter version of the COMP system 
    f(u,p,t) = twosqrtcomp(u,p,t);
    α₁ = 0.1; α₂ = 0.3; β₁ = 0.18; β₂ = 0.1; # standard parameters without ε (time-scale separation parameter)
    pf = vcat([α₁,α₂,β₁,β₂],[ε,σ]); # parameter vector
    u = zeros(2);
    g(u,p,t) = linear_diffusion ? [u[1]/2; u[2]/2] : idfunc; 
    pg = Float64[]; 
    Σ = I(2);
    process = WienerProcess(0.,u; save_everystep);
    StochSystem(f, pf, u, σ, g, pg, Σ, process)
end;