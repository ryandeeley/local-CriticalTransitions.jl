module CriticalTransitions

using Reexport
@reexport using DynamicalSystems
@reexport using OrdinaryDiffEq
@reexport using StochasticDiffEq
@reexport using DiffEqNoiseProcess
@reexport using LinearAlgebra
using Formatting, Dates, JLD2, HDF5, ProgressBars, ProgressMeter, DocStringExtensions
using IntervalRootFinding
using StaticArrays, ForwardDiff
using Symbolics
using Optim, Dierckx
using Printf, DrWatson, Dates, Statistics
using IntervalUnionArithmetic
using RandomNumbers.Xorshifts

include("utils.jl")
include("StochSystem.jl")

include("io/io.jl")
include("noiseprocesses/gaussian.jl")
include("systemanalysis/stability.jl")
include("systemanalysis/basinsofattraction.jl")
include("systemanalysis/basinboundary.jl")
include("trajectories/simulation.jl")
include("trajectories/transition.jl")
include("largedeviations/action.jl")
include("largedeviations/min_action_method.jl")
include("largedeviations/geometric_min_action_method.jl")

include("../systems/fitzhughnagumo.jl")
include("../systems/truscottbrindley_mod.jl")
include("../systems/truscottbrindley_mod_new.jl")
include("../systems/truscottbrindley_orig.jl")
include("../systems/truscottbrindley_orig1.jl")
include("../systems/rooth.jl")
include("../systems/stommel.jl")
include("../systems/rivals.jl")
include("../systems/double-well_potential.jl")

include("../dev/fhn_pathspace_sampling.jl")
include("../dev/symbolic_langevinmcmc.jl")
include("../dev/residence_times.jl")
include("../dev/edgetrack_ct.jl")
include("../dev/flexibletransitions.jl")
include("../dev/RateSys1.jl")
include("../dev/qp_along_curve.jl")
include("../dev/excursion.jl")

# Core types
export StochSystem, State

# Methods
export CoupledODEs, to_cds
export equilib, fixedpoints, basins, basinboundary
export simulate, relax
export transition, transitions
export transition_mf, transitions_mf
export semitransition, semitransitions
export langevinmcmc
export fw_integrand, fw_action, om_action, action, geometric_action
export min_action_method, geometric_min_action_method
export edgetracking, bisect_to_edge, bisect_to_edge2, attractor_mapper
export idfunc, idfunc!
export gauss
export drift, is_iip
export is_multiplicative, is_autonomous
export make_jld2, make_h5, sys_string, sys_info, intervals_to_box
export anorm, subnorm
export cov_inv_along_path

# Systems
export fitzhugh_nagumo, fitzhugh_nagumo!, fhn_εσ, fhn_ϵσ_backward, rotated_fhn_Aϵσ, fhn_maier_stein_form_ϵσ
export modifiedtruscottbrindley, modifiedtruscottbrindley!, modtb_αξσ, modtb_αξσ1, modtb_αξσ_backward
export modtb_αξσρ, modtbOU_αξγμσρ
export modifiedtruscottbrindley_OU, modifiedtruscottbrindley_OU!, modtbOU_αξγμσ, modtbOU_αξγμσ2, modtbOU_na_αξγY₀¹Y₀², modtbOU_alt_αξγ
export rampedmodifiedtruscottbrindley, modifiedtruscottbrindley!, rmodtb_ξvTtrTraσ
export originaltruscottbrindley, originaltruscottbrindley!, origtb_rσ
export rampedoriginaltruscottbrindley, rampedoriginaltruscottbrindley!, rorigtb_vTtrTraσ
export originaltruscottbrindley1, originaltruscottbrindley1!, origtb1_rσ
export rampedoriginaltruscottbrindley1, rampedoriginaltruscottbrindley1!, rorigtb1_vTtrTraσ
export rivals!, rivals, rivals_ϵσ
export rooth_smooth, stommel, cessi
export cov_inv_along_path
export dw_ησ

# Development
export transition2, transitions2
export residence_time2, residence_times2
export saddles_idx, repellers_idx, attractors_idx
export additive_idx, additive_idx!
export multiplicative_idx, multiplicative_idx!
export FitzHughNagumoSPDE, fhn_pathspace_sampling
export langevinmcmc_spde, symbolise_spde, stochastic_bridge, langevinmcmc_mf1, langevinmcmc_mf2 
export jacobian
export residence_time, residence_times, ResTimes, temporal, runandsavetimes, get_res_times
export exit_time, exit_times
export RateSystem, fL, stochtorate
export stochprocess
export qp_along_curve
export excursion, excursions
export hor_transition, hor_transitions

end # module CriticalTransitions
