r(t::Float64,ℓ::Float64,h::Float64) = (h/(2*exp(-2)))*exp(-ℓ/t);

function smoothramp(t::Float64,ℓ::Float64,h::Float64)
    if t ≤ 0 
        return 0
    elseif 0 < t ≤ ℓ/2
        return r(t,ℓ,h)
    elseif ℓ/2 ≤ t < ℓ
        return 2r(ℓ/2,ℓ,h)-r(ℓ-t,ℓ,h) 
    elseif t ≥ ℓ 
        return h
    end
end;

function proto(u,p,t)
    xₜ = u[1]; 
    ℓ,h = p[1]; 
    λₜ = smoothramp(t,ℓ,h); 
    return (xₜ+λₜ)^2-1. 
end;

function proto_autonomous(u,p,t)
    xₜ = u[1]; 
    λ = p[1][1]; 
    return (xₜ+λ)^2-1. 
end;

function proto_ℓσ(ℓ::Float64,σ::Float64;h::Float64 = 3.,pg::Vector{Float64}=Float64[],save_everystep::Bool = false,forward::Bool = true)
    f(u,p,t) = forward ? proto(u,p,t) : proto(u,p,t+ℓ);
    pf = [ℓ,h];
    u = zeros(1);
    g = idfunc;    
    Σ = I(1);
    process = WienerProcess(0.,u;save_everystep);
    StochSystem(f,pf,u,σ,g,pg,Σ,process)
end;

function proto_autonomous_λσ(λ::Float64,σ::Float64;pg::Vector{Float64}=Float64[],save_everystep::Bool = false)
    f = proto_autonomous;
    pf = [λ];
    u = zeros(1);
    g = idfunc;    
    Σ = I(1);
    process = WienerProcess(0.,u;save_everystep);
    StochSystem(f,pf,u,σ,g,pg,Σ,process)
end;