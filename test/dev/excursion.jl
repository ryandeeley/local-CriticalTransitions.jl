function excursion(sys::StochSystem, x_i::State, A::Function;
    rad_i=0.1,
    dt=0.01,
    tmax=1e3,
    solver=EM(),
    progress=true,
    cut_start=true,
    rad_dims=1:length(sys.u),
    kwargs...)

    ## first prescribe the condition for entering the desired region A

    condition1(u,t,integrator) = A(u)
    affect1!(integrator) = terminate!(integrator)
    cb_enter = DiscreteCallback(condition1, affect1!)

    sim = simulate(sys, x_i, dt=dt, tmax=tmax, solver=solver, callback=cb_enter, progress=progress, kwargs...)

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

    if success

        x_i_new = sim[:,end]; # new initial condition
        tmaxnew = tmax - simt[end]; # the remaining time prescribed to enter the neighbourhood of the point
        
        condition2(u,t,integrator) = subnorm(u - x_i; directions=rad_dims) < rad_i
        affect2!(integrator) = terminate!(integrator)
        cb_ball = DiscreteCallback(condition2, affect2!)         

        simnew = simulate(sys, x_i_new, dt=dt, tmax=tmaxnew, solver=solver, callback=cb_ball, progress=progress, kwargs...);

        successnew = true; 
        if simnew.t[end] == tmaxnew
            successnew = false
        end
        
        simboth = hcat(sim[:,1:end-1],Matrix(simnew));
        timeboth = vcat(simt,(simt[end] .+ simnew.t)[2:end]);

        return simboth, timeboth, successnew
        
    else
        return sim, simt, success
    end

end

function excursions(sys::StochSystem, x_i::State, A::Function, N=1;
    rad_i=0.1,
    dt=0.01,
    tmax=1e3,
    Nmax=1000,
    solver=EM(),
    cut_start=true,
    rad_dims=1:length(sys.u),
    savefile=nothing,
    showprogress::Bool=true)
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
        
        sim, simt, success = excursion(sys, x_i, A;
                    rad_i=rad_i, rad_dims=rad_dims, dt=dt, tmax=tmax,
                    solver=solver, progress=false, cut_start=cut_start)
        
        if success 
            
            if showprogress
                print("\rStatus: $(length(idx)+1)/$(N) excursions complete.")
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
