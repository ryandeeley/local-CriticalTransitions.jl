"""
    transition(sys::StochSystem, x_i::State, x_f::State; kwargs...)
Generates a sample transition from point `x_i` to point `x_f`.

This function simulates `sys` in time, starting from initial condition `x_i`, until entering a `length(sys.u)`-dimensional ball of radius `rad_f` around `x_f`.

## Keyword arguments
* `rad_i=0.1`: radius of ball around `x_i`
* `rad_f=0.1`: radius of ball around `x_f`
* `cut_start=true`: if `false`, returns the whole trajectory up to the transition
* `dt=0.01`: time step of integration
* `tmax=1e3`: maximum time when the simulation stops even `x_f` has not been reached
* `rad_dims=1:length(sys.u)`: the directions in phase space to consider when calculating the radii
  `rad_i` and `rad_f`. Defaults to all directions. To consider only a subspace of state space,
  insert a vector of indices of the dimensions to be included.
* `solver=EM()`: numerical solver. Defaults to Euler-Mayurama.
* `progress`: shows a progress bar with respect to `tmax`

## Output
`[path, times, success]`
* `path` (Matrix): transition path (size [dim × N], where N is the number of time points)
* `times` (Vector): time values (since start of simulation) of the path points (size N)
* `success` (bool): if `true`, a transition occured (i.e. the ball around `x_f` has been reached), else `false`
* `kwargs...`: keyword arguments passed to [`simulate`](@ref)

See also [`transitions`](@ref), [`simulate`](@ref).
"""
function transition(sys::StochSystem, x_i::State, x_f::State;
    rad_i=0.1,
    rad_f=0.1,
    dt=0.01,
    tmax=1e3,
    solver=EM(),
    progress=true,
    cut_start=true,
    rad_dims=1:length(sys.u),
    kwargs...)

    condition(u,t,integrator) = subnorm(u - x_f; directions=rad_dims) < rad_f
    affect!(integrator) = terminate!(integrator)
    cb_ball = DiscreteCallback(condition, affect!)

    sim = simulate(sys, x_i; dt, tmax, solver, callback=cb_ball, progress, kwargs...)

    success = true
    if sim.t[end] == tmax
        success = false
    end
    
    simt = sim.t
    if cut_start
        idx = size(sim)[2]
        dist = norm(sim[:,idx] - x_i)
        while dist > rad_i
            idx -= 1
            dist = norm(sim[:,idx] - x_i)
        end
        sim = sim[:,idx:end]
        simt = simt[idx:end]
    end

    sim, simt, success
end;

function hor_transition(sys::StochSystem, x_i::State, w::Float64;
    rad_i=0.01,
    dt=0.01,
    tmax=1e3,
    solver=EM(),
    progress=true,
    cut_start=true,
    rad_dims=1:length(sys.u),
    kwargs...)

    condition(u,t,integrator) = u[1] > w
    affect!(integrator) = terminate!(integrator)
    cb_wall = DiscreteCallback(condition, affect!)

    sim = simulate(sys, x_i; dt, tmax, solver, callback=cb_wall, progress, kwargs...)

    success = true
    if sim.t[end] == tmax
        success = false
    end

    simt = sim.t
    if cut_start
        idx = size(sim)[2]
        dist = norm(sim[:,idx] - x_i)
        while dist > rad_i
            idx -= 1
            dist = norm(sim[:,idx] - x_i)
        end
        sim = sim[:,idx:end]
        simt = simt[idx:end]
    end

    sim, simt, success
end;

function hor_transitions(sys::StochSystem, x_i::State, w::Float64, N=1;
    rad_i=0.01,
    dt=0.01,
    tmax=1e3,
    Nmax=1000,
    solver=EM(),
    cut_start=true,
    rad_dims=1:length(sys.u),
    savefile=nothing,
    showprogress::Bool=true,
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

        sim, simt, success = hor_transition(sys, x_i, w;
                    rad_i=rad_i, rad_dims=rad_dims, dt=dt, tmax=tmax,
                    solver=solver, progress=false, cut_start=cut_start, kwargs...)

        if success

            if showprogress
                print("\rStatus: $(length(idx)+1)/$(N) horizontal transitions complete.")
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

function semitransition(sys::StochSystem, x_i::State, x_f::State, inds::Vector;
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

    condition(u,t,integrator) = subnorm(u[inds] - x_f; directions=rad_dims) < rad_f
    affect!(integrator) = terminate!(integrator)
    cb_ball = DiscreteCallback(condition, affect!)

    u0 = zeros(length(sys.u)); u0[setdiff(1:length(sys.u),inds)] = x_i_rem; u0[inds] = x_i;

    sim = simulate(sys, u0; dt=dt, tmax=tmax, solver=solver, callback=cb_ball, progress=progress, kwargs...)

    success = true
    if sim.t[end] == tmax
        success = false
    end
    
    simt = sim.t
    if cut_start
        idx = size(sim)[2]
        dist = norm(sim[inds,idx] - x_i)
        while dist > rad_i && idx > 0
            idx -= 1
            dist = norm(sim[inds,idx] - x_i)
        end
        sim = sim[:,idx:end]
        simt = simt[idx:end]
    end

    sim, simt, success
end;

"""
    transitions(sys::StochSystem, x_i::State, x_f::State, N=1; kwargs...)
Generates an ensemble of `N` transition samples of `sys` from point `x_i` to point `x_f`.

This function repeatedly calls the [`transition`](@ref) function to efficiently generate an ensemble of transitions, which are saved to a file or returned as an array of paths. Multi-threading is enabled.

## Keyword arguments
* `rad_i=0.1`: radius of ball around `x_i`
* `rad_f=0.1`: radius of ball around `x_f`
* `cut_start=true`: if `false`, returns the whole trajectory up to the transition
* `Nmax`: number of attempts before the algorithm stops even if less than `N` transitions occurred.
* `dt=0.01`: time step of integration
* `tmax=1e3`: maximum time when the simulation stops even `x_f` has not been reached
* `rad_dims=1:length(sys.u)`: the directions in phase space to consider when calculating the radii
  `rad_i` and `rad_f`. Defaults to all directions. To consider only a subspace of state space,
  insert a vector of indices of the dimensions to be included.
* `solver=EM()`: numerical solver. Defaults to Euler-Mayurama
* `progress`: shows a progress bar with respect to `Nmax`
* `savefile`: if `nothing`, no data is saved to a file. To save to a file, see below.

See also [`transition`](@ref).

## Saving data to file
The `savefile` keyword argument allows saving the data to a `.jld2` or `.h5` file. To do so:
1. Create and open a file by typing `file = jld2open("filename.jld2", "a+")` or `file = h5open("filename.h5", "cw")`. This requires `JLD2.jl`/`HDF5.jl`; the convenience functions [`make_jld2`](@ref), [`make_h5`](@ref) provide this out of the box.
2. Pass the label `file` to the `savefile` argument of `transitions`.
3. Don't forget to `close(file)` at the end.

## Output
`[samples, times, idx, N_fail]`
* `samples` (Array of Matrices): sample paths. Each path i has size (dim × N_i), where N_i is the number of path points
* `times` (Array of Vectors): time values (since simulation start) of path points for each path
* `idx` (Array): list of sample indices i that produced a transition
* `N_fail` (Int): number of samples that failed to produce a transition

> An example script using `transitions` is available [here](https://github.com/reykboerner/CriticalTransitions.jl/blob/main/scripts/sample_transitions_h5.jl).
"""
function transitions(sys::StochSystem, x_i::State, x_f::State, N=1;
    rad_i=0.1,
    rad_f=0.1,
    dt=0.01,
    tmax=1e3,
    Nmax=1000,
    solver=EM(),
    cut_start=true,
    rad_dims=1:length(sys.u),
    savefile=nothing,
    showprogress::Bool=true,
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
        
        sim, simt, success = transition(sys, x_i, x_f;
                    rad_i=rad_i, rad_f=rad_f, rad_dims=rad_dims, dt=dt, tmax=tmax,
                    solver=solver, progress=false, cut_start=cut_start, kwargs...)
        
        if success 
            
            if showprogress
                print("\rStatus: $(length(idx)+1)/$(N) transitions complete.")
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

function semitransitions(sys::StochSystem, x_i::State, x_f::State, inds::Vector, N=1;
    rad_i=0.1,
    rad_f=0.1,
    dt=0.01,
    tmax=1e3,
    Nmax=1000,
    solver=EM(),
    cut_start=true,
    rad_dims=1:length(inds),
    savefile=nothing,
    showprogress::Bool=true,
    x_i_rem::State = [],
    kwargs...)
    """
    Generates N transition samples of sys from x_i to x_f.
    Supports multi-threading.
    rad_i:      ball radius around x_i
    rad_f:      ball radius around x_f
    cut_start:  if false, saves the whole trajectory up to the transition
    savefile:   if not nothing, saves data to a specified open .jld2 file
    """

    if isempty(x_i_rem)
    	x_i_rem = [zeros(length(sys.u)-length(inds)) for ii ∈ 1:Nmax];
    end

    samples, times, idx::Vector{Int64}, r_idx::Vector{Int64} = [], [], [], []

    iterator = showprogress ? tqdm(1:Nmax) : 1:Nmax

    Threads.@threads for j ∈ iterator
        
        sim, simt, success = semitransition(sys, x_i, x_f, inds;
                    rad_i=rad_i, rad_f=rad_f, rad_dims=rad_dims, dt=dt, tmax=tmax,
                    solver=solver, progress=false, cut_start=cut_start, x_i_rem = x_i_rem[j], kwargs...)
        
        if success 
            
            if showprogress
                print("\rStatus: $(length(idx)+1)/$(N) transitions complete.")
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


function semitransitions(sys::StochSystem, pn::Vector, x_i::State, x_f::State, inds::Vector, N=1;
    rad_i=0.1,
    rad_f=0.1,
    dt=0.01,
    tmax=1e3,
    Nmax=1000,
    solver=EM(),
    cut_start=true,
    rad_dims=1:length(inds),
    savefile=nothing,
    showprogress::Bool=true,
    kwargs...)
    """
    Generates N transition samples of sys from x_i to x_f.
    Supports multi-threading.
    rad_i:      ball radius around x_i
    rad_f:      ball radius around x_f
    cut_start:  if false, saves the whole trajectory up to the transition
    savefile:   if not nothing, saves data to a specified open .jld2 file
    """

    γ, μ, σ = pn;

    samples, times, idx::Vector{Int64}, r_idx::Vector{Int64} = [], [], [], []

    iterator = showprogress ? tqdm(1:Nmax) : 1:Nmax

    Threads.@threads for j ∈ iterator

        # only the initial condition needs to change

        u01 = μ[1] + randn()*σ[1]/√(2*γ[1]); u02 = μ[2] + randn()*σ[2]/√(2*γ[2]); # choosing initial conditions in the stationary regime
        process = OrnsteinUhlenbeckProcess([γ[1],γ[2],1,1], [μ[1],μ[2],0.,0.], [σ[1],σ[2],0.,0.], 0., [u01,u02,0.,0.]);

        sys = StochSystem(sys.f,sys.pf,sys.u,sys.σ,sys.g,sys.pg,sys.Σ,process); # wewrite the StochSystem each time
        
        sim, simt, success = semitransition(sys, x_i, x_f, inds;
                    rad_i=rad_i, rad_f=rad_f, rad_dims=rad_dims, dt=dt, tmax=tmax,
                    solver=solver, progress=false, cut_start=cut_start, kwargs...)
        
        if success 
            
            if showprogress
                print("\rStatus: $(length(idx)+1)/$(N) transitions complete.")
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

# in the following sysfunc is a function that returns a StochSystem with the given initial condition for the noise process
# at 23:36, I discovered that the erros lay in the definition of the callback conditions
function transition_mf(sysfunc::Function, process_init::State, x_i::State, x_f::State, inds::Vector;
    rad_i=0.01,
    rad_f=0.01,
    dt=0.01,
    tmax=1e3,
    Ttrans::Float64 = 100.,
    perc::Float64 = 1.,
    solver=EM(),
    progress=true,
    rad_dims=1:length(inds),
    x_i_rem = zeros(length(sysfunc(process_init).u)-length(inds)),
    dispfreq::Int64 = 1000000,
    kwargs...)

    dim = length(sysfunc(process_init).u);

    # exiting the neighbourhood callback condition
    condition_ex(u,t,integrator) = subnorm(integrator.u[inds] - x_i; directions=rad_dims) > rad_i
    affect_ex!(integrator) = terminate!(integrator)
    cb_exit = DiscreteCallback(condition_ex, affect_ex!)

    # either reentering B(R) or entering B(L) callback condition
    condition_en(u,t,integrator) = subnorm(integrator.u[inds] - x_i; directions=rad_dims) < rad_i || subnorm(integrator.u[inds] - x_f; directions=rad_dims) < rad_f 
    affect_en!(integrator) = terminate!(integrator)
    cb_enter = DiscreteCallback(condition_en, affect_en!)

    sys = sysfunc(process_init);
    u0 = zeros(length(sys.u)); u0[setdiff(1:length(sys.u),inds)] = x_i_rem; u0[inds] = x_i;

    #T = zeros(2); T[1] = tmax*perc; 
    # first entry holds the total time prescribed for the most recent simulation
    # second entry holds the accumulative total run time 

    #sim = simulate(sys, u0; dt, tmax = T[1], solver, callback=cb_exit, progress, kwargs...)
    p_tmax = tmax*perc;
    sim = simulate(sys, u0; dt, tmax = p_tmax, solver, callback=cb_exit, progress, save_everystep = false, save_start = false, kwargs...)

    #T[2] = sim.t[end]; # the total time of the run 
    T = sim.t[end]; # the total time of the run
    u0 = sim.u[end]; # where the run terminated, which becomes the new initial condition
    process_init = sys.process.u[end]; sys = sysfunc(process_init);

    #println("State, noise process initial condition: $(u0), $(sys.process.u[end])")

    # for correlated noise processes, we need to transfer the initial condition to the process
    
        # while T[2] < tmax && sim.t[end] ≥ T[1] # i.e. the simulation just completed without achieving any of the callback conditions
        #     # i.e. simulate again until the callback condition is met
        #     T[1] = min(perc*tmax,tmax-T[2]);
        #     sim = simulate(sys, u0; dt, tmax = T[1], solver, callback=cb_exit, progress, kwargs...)
        #     T[2] += sim.t[end]; # the total time of the run 
        #     u0 = sim.u[end]; # where the run terminated, which becomes the new initial condition
        #     process_init = sys.process.u[end]; sys = sysfunc(process_init);
        # end

    exit = true; reentered = false; # if T < tmax we must have exited the neighbourhood 
    success = false; 

    counter = 1; 

    #while T[2] < tmax && (reentered||exit) #subnorm(u0 - x_f; directions=rad_dims) ≥ rad_f
    while T < tmax && (reentered||exit) #&& counter[2] < 5 #subnorm(u0 - x_f; directions=rad_dims) ≥ rad_f
        if progress && mod(counter,dispfreq)==0
            print("\rAttempt $(counter[1]) to transition; the accumulated run time is T = $(T)")
        end 
        counter += 1; 
        if reentered
            #print("\rI entered the re-entered section: time $(counter[3])"); counter[3] += 1; 
            #T[1] = min(perc*tmax,tmax-T[2]);
            #sim = simulate(sys, u0; dt, tmax = T[1], solver, callback=cb_exit, progress, kwargs...); # overwrite the old solution
            p_tmax = min(tmax*perc,tmax-T);
            sim = simulate(sys, u0; dt, tmax = p_tmax, solver, callback=cb_exit, progress, save_everystep = false, save_start = false, kwargs...); # overwrite the old solution
            #T[2] += sim.t[end]; # the new overall elapsed time
            T += sim.t[end]; # the new overall elapsed time
            u0 = sim.u[end]; # where the run terminated, which becomes the new initial condition
            process_init = sys.process.u[end]; sys = sysfunc(process_init);
            # while T[2] < tmax && sim.t[end] ≥ T[1] # i.e. the simulation just completed without achieving any of the callback conditions
            #     # i.e. simulate again until the callback condition is met
            #     T[1] = minimum(perc*tmax,tmax-T[2]);
            #     sim = simulate(sys, u0; dt, tmax = T[1], solver, callback=cb_exit, progress, kwargs...)
            #     T[2] += sim.t[end]; # the total time of the run 
            #     u0 = sim.u[end]; # where the run terminated, which becomes the new initial condition
            #     process_init = sys.process.u[end]; sys = sysfunc(process_init);
            # end
            exit = true; reentered = false;  
        else #i.e. exit must be true 
            #println("I entered the exit section: time $(counter[2])"); counter[2] += 1; 
            #T[1] = min(perc*tmax,tmax-T[2]);
            #sim = simulate(sys, u0; dt, tmax = T[1], solver, callback=cb_enter, progress, kwargs...); # overwrite the old solution
            #print("\rState, noise process initial condition: $(u0), $(sys.process.u[end])")
            sim = simulate(sys, u0; dt, tmax = min(Ttrans,tmax-T), solver, callback=cb_enter, progress, kwargs...); # overwrite the old solution
            #T[2] += sim.t[end]; # the new overall elapsed time
            T += sim.t[end]; # the new overall elapsed time
            u0 = sim.u[end]; # where the run terminated, which becomes the new initial condition
            process_init = sys.process.u[end]; sys = sysfunc(process_init);
	        # while T[2] < tmax && sim.t[end] ≥ T[1] # i.e. the simulation just completed without achieving any of the callback conditions
            #     # i.e. simulate again until the callback condition is met
            #     T[1] = minimum(perc*tmax,tmax-T[2]);
            #     sim = simulate(sys, u0; dt, tmax = T[1], solver, callback=cb_exit, progress, kwargs...)
            #     T[2] += sim.t[end]; # the total time of the run 
            #     u0 = sim.u[end]; # where the run terminated, which becomes the new initial condition
            #     process_init = sys.process.u[end]; sys = sysfunc(process_init);
            # end
            if subnorm(u0[inds] - x_i; directions=rad_dims) < rad_i # we have reentered the neighbourhood of the initial attractor
                reentered = true; exit = false;
	        elseif subnorm(u0[inds] - x_f; directions=rad_dims) < rad_f # we have reentered the neighbourhood of the final attractor
                reentered = false; exit = false; # i.e. the transition is complete
                success = true; 
            end
        end
        #println("\r$(T[1])")
    end

    # translated_times = (T[2]-sim.t[end]) .+ sim.t;
    translated_times = (T-sim.t[end]) .+ sim.t;
    statevars = reduce(hcat,[[sim.u[ii][jj] for ii in 1:length(sim.u)] for jj in 1:dim])';

    statevars, translated_times, success

end

# second version: utilising multiple dispatch
# for cases where the value of the noise process is independent from its previous value, and you are constraining all coordinates  
function transition_mf(sys::StochSystem, x_i::State, x_f::State;
    rad_i=0.01,
    rad_f=0.01,
    dt=0.01,
    tmax=1e3,
    Ttrans::Float64 = 100.,
    perc::Float64 = 1.,
    solver=EM(),
    progress=true,
    rad_dims=1:length(x_i),
    dispfreq::Int64 = 1000000,
    kwargs...)

    dim = length(x_i);

    # exiting the neighbourhood callback condition
    condition_ex(u,t,integrator) = subnorm(integrator.u - x_i; directions=rad_dims) > rad_i
    affect_ex!(integrator) = terminate!(integrator)
    cb_exit = DiscreteCallback(condition_ex, affect_ex!)

    # either reentering B(R) or entering B(L) callback condition
    condition_en(u,t,integrator) = subnorm(integrator.u - x_i; directions=rad_dims) < rad_i || subnorm(integrator.u - x_f; directions=rad_dims) < rad_f 
    affect_en!(integrator) = terminate!(integrator)
    cb_enter = DiscreteCallback(condition_en, affect_en!)

    u0 = x_i;

    p_tmax = tmax*perc;
    sim = simulate(sys, u0; dt, tmax = p_tmax, solver, callback=cb_exit, progress, save_everystep = false, save_start = false, kwargs...)

    T = sim.t[end]; # the total time of the run
    u_in,u_out = [sim.u[end-1],sim.u[end]]; # where the run terminated, which becomes the new initial condition

    exit = true; reentered = false; # if T < tmax we must have exited the neighbourhood 
    success = false; 

    counter = 1; 

    while T < tmax && (reentered||exit)
        if progress && mod(counter,dispfreq)==0
            print("\rAttempt $(counter[1]) to transition; the accumulated run time is T = $(T)")
        end 
        counter += 1; 
        if reentered
            p_tmax = min(tmax*perc,tmax-T);
            sim = simulate(sys, u0; dt, tmax = p_tmax, solver, callback=cb_exit, progress, save_everystep = false, save_start = false, kwargs...); # overwrite the old solution
            T += sim.t[end]; # the new overall elapsed time
            u_in, u_out = [sim.u[end-1],sim.u[end]]; # where the run terminated, which becomes the new initial condition
            exit = true; reentered = false;  
        else #i.e. exit must be true 
            sim = simulate(sys, u_out; dt, tmax = min(Ttrans,tmax-T), solver, callback=cb_enter, progress, kwargs...); # overwrite the old solution
            T += sim.t[end]; # the new overall elapsed time
            u0 = sim.u[end]; # where the run terminated, which becomes the new initial condition
            if subnorm(u0 - x_i; directions=rad_dims) < rad_i # we have reentered the neighbourhood of the initial attractor
                reentered = true; exit = false;
	        elseif subnorm(u0 - x_f; directions=rad_dims) < rad_f # we have reentered the neighbourhood of the final attractor
                reentered = false; exit = false; # i.e. the transition is complete
                success = true; 
            end
        end
    end

    translated_times = (T-sim.t[end]) .+ sim.t; 
    statevars = reduce(hcat,[[sim.u[ii][jj] for ii in 1:length(sim.u)] for jj in 1:dim])';

    if success # if the system transitioned, then we concatenate the state position just before it exited the neighbourhood around x_i
        translated_times = vcat(translated_times,[T+dt]);
        statevars = hcat(u_in,statevars);
    end

    statevars, translated_times, success

end

function transitions_mf(sysfunc::Function, process_init::State, x_i::State, x_f::State, inds::Vector, N=1;
    rad_i=0.01,
    rad_f=0.01,
    tmax=1.e3,
    dt=0.01,
    Ttrans=1.e2,
    perc::Float64=1.,
    Nmax=1000,
    solver=EM(),
    rad_dims=1:length(inds),
    individual_save=false,
    savepath = nothing,
    showprogress::Bool = true,
    x_i_rem = zeros(length(sysfunc(process_init).u)-length(inds)),
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
        
        sim, simt, success = transition_mf(sysfunc, process_init, x_i, x_f, inds;
                    rad_i, rad_f, rad_dims, tmax, dt, Ttrans, perc,
                    solver, progress=false, x_i_rem, kwargs...)
        
        if success 
            
            if showprogress
                print("\rStatus: $(length(idx)+1)/$(N) (memory-friendly) transitions complete.")
            end

            if individual_save 
                safesave(savepath,Dict("data"=>[sim,simt]))
            else
                push!(samples, sim);
                push!(times, simt);
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

# second version: utilising multiple dispatch
# for cases where the value of the noise process is independent from its previous value, and you are constraining all coordinates  
function transitions_mf(sys::StochSystem, x_i::State, x_f::State, N::Int64 = 1;
    rad_i=0.01,
    rad_f=0.01,
    tmax=1.e3,
    dt=0.01,
    Ttrans=1.e2,
    perc::Float64=1.,
    Nmax=1000,
    solver=EM(),
    rad_dims=1:length(x_i),
    individual_save=false,
    savepath = nothing,
    showprogress::Bool = true,
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
                    rad_i, rad_f, rad_dims, tmax, dt, Ttrans, perc,
                    solver, progress=false, kwargs...)
        
        if success 
            
            if showprogress
                print("\rStatus: $(length(idx)+1)/$(N) (memory-friendly) transitions complete.")
            end

            if individual_save 
                safesave(savepath,Dict("data"=>[sim,simt]))
            else
                push!(samples, sim);
                push!(times, simt);
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