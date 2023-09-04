"""
Dynamical systems specification file
"""

# fitzhugh_nagumo model

"""
    fitzhugh_nagumo!(du, u, p, t)
In-place definition of the FitzHugh-Nagumo system.

See also [`fitzhugh_nagumo`](@ref).
"""
function fitzhugh_nagumo!(du, u, p, t)
    x, y = u
    ε, β, α, γ, κ, I = p[1]

    du[1] = (-α*x^3 + γ*x - κ*y + I)/ε
    du[2] = -β*y + x
end

"""
    fitzhugh_nagumo(u, p, t)
Out-of-place definition of the FitzHugh-Nagumo system.

See also [`fitzhugh_nagumo!`](@ref).
"""
function fitzhugh_nagumo(u,p,t)
    x, y = u
    ε, β, α, γ, κ, I = p[1]

    dx = (-α*x^3 + γ*x - κ*y + I)/ε
    dy = -β*y + x

    SA[dx, dy]
end

function rotated_fitzhugh_nagumo(u,p,t)
    x, y = u
    pf, A = p[1]

    xₜ,yₜ = inv(A)*[x;y]; # inverting the transformation 

    ε, β, α, γ, κ, I = pf;
    dx = A[1,1]*((-α*xₜ^3 + γ*xₜ - κ*yₜ + I)/ε) + A[1,2]*(-β*yₜ + xₜ);
    dy = A[2,1]*((-α*xₜ^3 + γ*xₜ - κ*yₜ + I)/ε) + A[2,2]*(-β*yₜ + xₜ);

    SA[dx,dy]
end

# For backwards compatibility
FitzHughNagumo(u,p,t) = fitzhugh_nagumo(u,p,t)
FitzHughNagumo!(u,p,t) = fitzhugh_nagumo!(u,p,t)


"""
    fhn_εσ(ε,σ)
A shortcut command for returning a StochSystem of the FitzHugh Nagumo system in a default setup with additive isotropic noise. 
    
This setup fixes the parameters β = 3, α =  γ = κ = 1, I = 0 and leaves the value of the time-scale parameter ε as a function argument. The prescribed noise process is additive and isotropic: the variables are peturbed by independently drawn identical Gaussian white noise realisations, with standard deviation σ (the other function argument).
"""
function fhn_εσ(ε, σ) # a convenient two-parameter version of the FitzHugh Nagumo system 
    # defining the StochSystem
    f(u,p,t) = fitzhugh_nagumo(u,p,t);
    β = 3; α = γ = κ = 1; I = 0; # standard parameters without ε (time-scale separation parameter)
    pf_wo_ε = [β, α, γ, κ, I]; # parameter vector without ε
    u = zeros(2);
    g = idfunc;
    pg = nothing; 
    Σ = [1 0; 0 1];
    process = WienerProcess(0.,u);
    StochSystem(f, vcat([ε], pf_wo_ε), u, σ, g, pg, Σ, process)
end;

function fhn_εσ_backward(ε, σ) # a convenient two-parameter version of the FitzHugh Nagumo system 
    # defining the StochSystem
    f(u,p,t) = -fitzhugh_nagumo(u,p,t);
    β = 3; α = γ = κ = 1; I = 0; # standard parameters without ε (time-scale separation parameter)
    pf_wo_ε = [β, α, γ, κ, I]; # parameter vector without ε
    dim = 2;
    g = idfunc;
    pg = nothing; 
    Σ = [1 0; 0 1];
    process = "WhiteGauss";
    StochSystem(f, vcat([ε], pf_wo_ε), dim, σ, g, pg, Σ, process)
end;

function rotated_fhn_Aεσ(A, ε, σ; t₀ = 1) # a convenient two-parameter version of the FitzHugh Nagumo system
    # defining the StochSystem
    f(u,p,t) = t₀.*rotated_fitzhugh_nagumo(u,p,t);
    u = zeros(2);
    g = idfunc;
    pg = nothing; 
    Σ = [1 0; 0 1];
    process = WienerProcess(0.,u);
    ## defining the system parameters
    β = 3; α = γ = κ = 1; I = 0; # standard parameters without ε (time-scale separation parameter)
    pf_wo_ε = [β, α, γ, κ, I]; # parameter vector without ε
    pf = vcat([ε], pf_wo_ε);
    StochSystem(f, [pf,A], u, σ, g, pg, Σ, process)
end;

function fhn_maier_stein_form_εσ(ε,σ)
    m = (-(1+3*ε)+√((1+3*ε)^2-4*ε))/(2*ε);
    A = [1 m; 0 1];
    t₀ = ε/(1+ε*m);
    rotated_fhn_Aεσ(A,ε,σ;t₀)
end