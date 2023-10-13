    """
Dynamical systems specification file
"""

# modified Truscott-Brindley system

"""
    modifiedtruscottbrindley!(du, u, p, t)
In-place definition of the modified Truscott-Brindley system. 

See also [`modifiedtruscottbrindley`](@ref).
"""
function modifiedtruscottbrindley!(du, u, p, t)
    P, Z = u
    α, β, γ, P₁, Z₁, ξ = p[1]

    du[1] = P₁*(α*(P/P₁)*(1-β*(P/P₁))-γ*(Z/Z₁)*(P/P₁)^2/(1+(P/P₁)^2));
    du[2] = ξ*Z₁* ((Z/Z₁)*(P/P₁)^2/(1+(P/P₁)^2)-(Z/Z₁)^2);
end

"""
    modifiedtruscottbrindley(u, p, t)
Out-of-place definition of the modified Truscott-Brindley system. 

See also [`modifiedtruscottbrindley!`](@ref).
"""
function modifiedtruscottbrindley(u,p,t)
    P, Z = u
    α, β, γ, P₁, Z₁, ξ = p[1]

    dP = (P₁/ξ)*(α*(P/P₁)*(1-β*(P/P₁))-γ*(Z/Z₁)*(P/P₁)^2/(1+(P/P₁)^2));
    dZ = Z₁*((Z/Z₁)*(P/P₁)^2/(1+(P/P₁)^2)-(Z/Z₁)^2);

    SVector{2}(dP, dZ)
end

"""
    modifiedtruscottbrindley_OU!(u, p, t)
In-place definition of the modified Truscott-Brindley system with correlated noise. 

See also [`modifiedtruscottbrindley_OU`](@ref).
"""
function modifiedtruscottbrindley_OU!(du,u,p,t)
    P, Z, X₁, X₂ = u
    α, β, γ, P₁, Z₁, ξ, ρ = p[1]
	
    f(P,Z) = (1/ξ)*(P₁*(α*(P/P₁)*(1-β*(P/P₁))-γ*(Z/Z₁)*(P/P₁)^2/(1+(P/P₁)^2)));
    g(P,Z) = Z₁* ((Z/Z₁)*(P/P₁)^2/(1+(P/P₁)^2)-(Z/Z₁)^2);

    Aρ = [√((1+ρ)/2) √((1-ρ)/2); √((1+ρ)/2) -√((1-ρ)/2)];

    du[1], du[2] = [f(P,Z); g(P,Z)] + [P/ξ 0; 0 Z]*Aρ*[X₁; X₂];
    du[3], du[4] = [0; 0];
end

"""
    modifiedtruscottbrindley(u, p, t)
Out-of-place definition of the modified Truscott-Brindley system with correlated noise. 

See also [`modifiedtruscottbrindley_OU!`](@ref).
"""
function modifiedtruscottbrindley_OU_new(u,p,t)
    P, Z, X₁, X₂ = u
    α, β, γ, P₁, Z₁, ξ = p[1]

    f(P,Z) = (1/ξ)*(P₁*(α*(P/P₁)*(1-β*(P/P₁))-γ*(Z/Z₁)*(P/P₁)^2/(1+(P/P₁)^2)));
    g(P,Z) = Z₁* ((Z/Z₁)*(P/P₁)^2/(1+(P/P₁)^2)-(Z/Z₁)^2);

    dP, dZ = [f(P,Z); g(P,Z)] .+ [P*X₁/√ξ; Z*X₂];

    SVector{4}(dP, dZ, 0., 0.)
end;

"""
    modtb_αξσρ(α, ξ, σ)
A shortcut command for returning a StochSystem of the modified Truscott-Brindley system in a default setup with multiplicative anisotropic noise. 
    
This setup fixes the parameters β = 5/112, γ = 112/2.3625, P₁ = β, Z₁ = 5/6 and leaves the values of the parameters α and ξ as function arguments. The prescribed noise process is multiplicative and anisotropic: the first variable is peturbed by Gaussian white noise realisations that are multiplied by the variable's current value; the second variable has no stochastic component. The noise strength σ and correlation coefficient ρ are left as the remaining function arguments.
"""
function modtb_αξσρ(α, ξ, σ, ρ; save_everystep = false) # a convenient four-parameter version of the modifiedtruscottbrindley system 
    f(u,p,t) = modifiedtruscottbrindley(u,p,t);
    β = 5/112; γ = 112/(45*0.0525); P₁ = β; Z₁ = 5/6; # standard parameters without α (growth rate) and ξ (time-scale separation)
    pf_wo_αξ = [β, γ, P₁, Z₁]; # parameters vector without α or ξ
    u = zeros(2);
    g(u,p,t) = [1/√ξ 0; 0 1.]*multiplicative_idx(u,p,t,[true,true]);
    pg = []; 
    Σ = [1. ρ; ρ 1.];
    process = WienerProcess(0., u; save_everystep);
    StochSystem(f, Float64[[α];pf_wo_αξ;[ξ]], u, σ, g, pg, Σ, process)
end;

"""
    modtbOU_αξσ(α, ξ, σ)
A shortcut command for returning a StochSystem of the modified Truscott-Brindley system with correlated in a default setup with multiplicative anisotropic noise. 
    
This setup fixes the parameters β = 5/112, γ = 112/2.3625, P₁ = β, Z₁ = 5/6 and leaves the values of the parameters α and ξ as function arguments. The prescribed noise process is multiplicative and anisotropic: the first variable is peturbed by Gaussian white noise realisations that are multiplied by the variable's current value; the second variable has no stochastic component. The noise strength σ is left as the remaining function argument.
"""
function modtbOU_αξγμσρ(α, ξ, γ, μ, σ, ρ; σₙ = 1., u0 = [0.,0.], stat_dist = false) # a convenient five-parameter version of the modifiedtruscottbrindley system 
    f(u,p,t) = modifiedtruscottbrindley_OU_new(u,p,t);
    β = 5/112; γ1 = 112/(45*0.0525); P₁ = β; Z₁ = 5/6; # standard parameters without α (growth rate) and ξ (time-scale separation)
    pf_wo_αξ = [β, γ1, P₁, Z₁]; # parameters vector without α or ξ
    u = zeros(4);
    g(u,p,t) = [0 0; 0 0; √((1+ρ)/2) √((1-ρ)/2); √((1+ρ)/2) -√((1-ρ)/2)];
    pg = []; 
    Σ = I(2);
    if stat_dist # prescribing initial conditions for the noise process from the stationary distribution
        u01 = μ[1] + randn()*σₙ*σ[1]/√(2*γ[1]); u02 = μ[2] + randn()*σₙ*σ[2]/√(2*γ[2]); # choosing initial conditions in the stationary regime
        u0 = [u01,u02]; # overwriting the initial condition for the noise process
    end
    process = OrnsteinUhlenbeckProcess(γ, μ, σₙ.*σ, 0., u0);
    StochSystem(f, Float64[[α];pf_wo_αξ;[ξ]], u, 1., g, pg, Σ, process)
end;