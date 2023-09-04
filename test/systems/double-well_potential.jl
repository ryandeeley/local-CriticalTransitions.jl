# double-well potential 

function  doublewell(u,p,t)
    x, y = u

    dx = 2y - (x+y)^3;
    dy = 2x - (x+y)^3;
    
    SA[dx,dy]
end

function dw_ησ(η::Float64,σ::Float64) # a convenient two-parameter version of the double-well potential system 
    f(u,p,t) = doublewell(u,p,t);
    u = zeros(2);
    g(u,p,t) = [1.;1.];
    pf = pg = nothing; 
    Σ = [1. 0.; 0. η];
    process = WienerProcess(0., u);
    StochSystem(f, pf, u, σ, g, pg, Σ, process)
end;