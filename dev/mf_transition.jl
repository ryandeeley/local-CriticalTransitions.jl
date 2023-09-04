function transition_mf(sys::StochSystem, x_i::State, x_f::State;
    rad_i=0.1,
    rad_f=0.1,
    dt=0.01,
    tmax=1e3,
    solver=EM(),
    progress=true,
    cut_start=true,
    rad_dims=1:length(sys.u),
    kwargs...)

    # exiting the neighbourhood callback condition
    condition(u,t,integrator) = subnorm(u - x_i; directions=rad_dims) > rad_i
    affect!(integrator) = terminate!(integrator)
    cb_exit = DiscreteCallback(condition, affect!)

    # either reentering B(R) or entering B(L) callback condition
    condition(u,t,integrator) = subnorm(u - x_i; directions=rad_dims) < rad_i || subnorm(u - x_f; directions=rad_dims) < rad_f 
    affect!(integrator) = terminate!(integrator)
    cb_enter = DiscreteCallback(condition, affect!)

    sim = simulate(sys, x_i; dt, tmax, solver, callback=cb_exit, progress, kwargs...)

    T = sim.t[end]; 
    exit = true; # if T < tmax we must have exited the neighbourhood 
    success = false; 
    
    while T < tmax && (exit || reentered) 
        if exit 
            sim = simulate(sys, x_i; dt, tmax = tmax - T, solver, callback=cb_enter, progress, kwargs...); # overwrite the old solution
            T += sim.t[end]; # the new overall elapsed time
            if T < tmax && subnorm(sim.u[:,end] - x_i; directions=rad_dims) < rad_i # we have reentered the neighbourhood of the initial attractor
                reentered = true; exit = false;
            elseif T < tmax && subnorm(sim.u[:,end] - x_f; directions=rad_dims) < rad_f # we have reentered the neighbourhood of the final attractor
                reentered = false; exit = false; # i.e. the transition is complete
                success = true; 
            end
        elseif reentered
            sim = simulate(sys, x_i; dt, tmax = tmax - T, solver, callback=cb_exit, progress, kwargs...); # overwrite the old solution
            T += sim.t[end]; # the new overall elapsed time
            exit = true; reentered = false;  
        end
    end

    sim.u, sim.t, success

end