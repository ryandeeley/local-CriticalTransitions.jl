function transition_mf(sys::StochSystem, x_i::State, x_f::State;
    rad_i=0.01,
    rad_f=0.01,
    dt=0.01,
    tmax=1e3,
    solver=EM(),
    progress=true,
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
    exit = true; reentered = false; # if T < tmax we must have exited the neighbourhood 
    success = false; 
    
    counter = 1; 

    while T < tmax && (exit || reentered) 
        if progress && mod(counter,1000)==0
            print("\rAttempt $(counter) to transition.")
        end
        counter += 1;
        if exit 
            sim = simulate(sys, x_i; dt, tmax = tmax - T, solver, callback=cb_enter, progress, kwargs...); # overwrite the old solution
            T += sim.t[end]; # the new overall elapsed time
            if T < tmax && subnorm(sim[:,end] - x_i; directions=rad_dims) < rad_i # we have reentered the neighbourhood of the initial attractor
                reentered = true; exit = false;
            elseif T < tmax && subnorm(sim[:,end] - x_f; directions=rad_dims) < rad_f # we have reentered the neighbourhood of the final attractor
                reentered = false; exit = false; # i.e. the transition is complete
                success = true; 
            end
        elseif reentered
            sim = simulate(sys, x_i; dt, tmax = tmax - T, solver, callback=cb_exit, progress, kwargs...); # overwrite the old solution
            T += sim.t[end]; # the new overall elapsed time
            exit = true; reentered = false;  
        end
    end

    sim, sim.t, success

end

function semitransition_mf(sys::StochSystem, x_i::State, x_f::State, inds::Vector;
    rad_i=0.01,
    rad_f=0.01,
    dt=0.01,
    tmax=1e3,
    solver=EM(),
    progress=true,
    cut_start=true,
    rad_dims=1:length(inds),
    x_i_rem = zeros(length(sys.u)-length(inds)),
    kwargs...)

    # exiting the neighbourhood callback condition
    condition(u,t,integrator) = subnorm(u[inds] - x_i; directions=rad_dims) > rad_i
    affect!(integrator) = terminate!(integrator)
    cb_exit = DiscreteCallback(condition, affect!)

    # either reentering B(R) or entering B(L) callback condition
    condition(u,t,integrator) = subnorm(u[inds] - x_i; directions=rad_dims) < rad_i || subnorm(u[inds] - x_f; directions=rad_dims) < rad_f 
    affect!(integrator) = terminate!(integrator)
    cb_enter = DiscreteCallback(condition, affect!)

    u0 = zeros(length(sys.u)); u0[setdiff(1:length(sys.u),inds)] = x_i_rem; u0[inds] = x_i;

    sim = simulate(sys, u0; dt, tmax, solver, callback=cb_exit, progress, kwargs...)

    T = sim.t[end]; # the time of the run 
    u0 = sim[:,end]; # where the run terminated, which becomes the new initial condition 
    exit = true; reentered = false; # if T < tmax we must have exited the neighbourhood 
    success = false; 
    
    counter = 1; 

    while T < tmax && (exit || reentered) 
        if progress && mod(counter,1000)==0
            print("\rAttempt $(counter) to transition.")
        end
        counter += 1;
        if exit 
            sim = simulate(sys, x_i; dt, tmax = tmax - T, solver, callback=cb_enter, progress, kwargs...); # overwrite the old solution
            T += sim.t[end]; # the new overall elapsed time
            if T < tmax && subnorm(sim[:,end] - x_i; directions=rad_dims) < rad_i # we have reentered the neighbourhood of the initial attractor
                reentered = true; exit = false;
            elseif T < tmax && subnorm(sim[:,end] - x_f; directions=rad_dims) < rad_f # we have reentered the neighbourhood of the final attractor
                reentered = false; exit = false; # i.e. the transition is complete
                success = true; 
            end
        elseif reentered
            sim = simulate(sys, x_i; dt, tmax = tmax - T, solver, callback=cb_exit, progress, kwargs...); # overwrite the old solution
            T += sim.t[end]; # the new overall elapsed time
            exit = true; reentered = false;  
        end
    end

    sim, sim.t, success

end


function transitions_mf(sys::StochSystem, x_i::State, x_f::State, N=1;
    rad_i=0.01,
    rad_f=0.01,
    dt=0.01,
    tmax=1e3,
    solver=EM(),
    progress=true,
    rad_dims=1:length(sys.u),
    kwargs...)
    """
    Generates N transition samples of sys from x_i to x_f.
    Supports multi-threading.
    rad_i:      ball radius around x_i
    rad_f:      ball radius around x_f
    cut_start:  if false, saves the whole trajectory up to the transition
    savefile:   if not nothing, saves data to a specified open .jld2 file
    """

    samples, times, idx::Vector{Int64}, r_idx::Vector{Int64} = [], [], [], []

    iterator = showprogress ? tqdm(1:Nmax) : 1:Nmax

    Threads.@threads for j ∈ iterator
        
        sim, simt, success = transition_mf(sys, x_i, x_f;
                    rad_i, rad_f, rad_dims, dt, tmax,
                    solver, progress=false, kwargs...)
        
        if success 
            
            if showprogress
                print("\rStatus: $(length(idx)+1)/$(N) (memory-friendly) transitions complete.")
            end

            if savefile == nothing
                push!(samples, sim);
                push!(times, simt);
            else # store or save in .jld2/.h5 file
                write(savefile, "paths/path "*string(j), sim)
                write(savefile, "times/times "*string(j), simt)
            end
        
            push!(idx, j)

            if length(idx) > max(1, N - Threads.nthreads())
                break
            else
                continue
            end
        else
            push!(r_idx, j)
        end
    end

    samples, times, idx, length(r_idx)    
end;