#include("../StochSystem.jl")
#include("../utils.jl")

"""
    fw_action(sys::StochSystem, path, time; kwargs...)
Calculates the Freidlin-Wentzell action of a given `path` with time points `time` in a 
drift field specified by the deterministic dynamics of `sys`.

The path must be a `(D x N)` matrix, where `D` is the dimensionality of the system `sys` and
`N` is the number of path points. The `time` array must have length `N`.

Returns a single number, which is the value of the action functional

``S_T[\\phi_t] = \\frac{1}{2} \\int_0^T || \\dot \\phi_t - b(\\phi_t) ||^2_Q dt``

where ``\\phi_t`` denotes the path in state space, ``b`` the drift field, and ``T`` the
total time of the path. The subscript ``Q`` refers to the
generalized norm ``||a||_Q^2 := \\langle a, Q^{-1} b \\rangle`` (see [`anorm`](@ref)). Here
``Q`` is the noise covariance matrix `sys.Σ`.

## Keyword arguments
* `cov_inv = nothing`: Inverse of the covariance matrix ``\\Sigma``.
  If `nothing`, ``\\Sigma^{-1}`` is computed automatically.
"""
function fw_action(sys::StochSystem, path, time; cov_inv=nothing)
    # Inverse of covariance matrix
    (cov_inv == nothing) ? A = inv(sys.Σ) : A = cov_inv
    
    # Compute action integral
    integrand = fw_integrand(sys, path, time, A)
    S = 0
    for i in 1:(size(path, 2) - 1)
        S += (integrand[i+1] + integrand[i])/2 * (time[i+1]-time[i])
    end
    S/2
end;

"""
    om_action(sys::StochSystem, path, time; kwargs...)
Calculates the Onsager-Machlup action of a given `path` with time points `time` for the
drift field `sys.f` at noise strength `sys.σ`.

The path must be a `(D x N)` matrix, where `D` is the dimensionality of the system `sys` and
`N` is the number of path points. The `time` array must have length `N`.

Returns a single number, which is the value of the action functional

``I^{\\sigma}_T[\\phi_t] = \\frac{1}{2} \\int_0^T \\left( || \\dot \\phi_t - b(\\phi_t) ||^2_Q +
\\frac{\\sigma^2}{2} \\div(b) \\right) \\, dt``

where ``\\phi_t`` denotes the path in state space, ``b`` the drift field, ``T`` the total
time of the path, and ``\\sigma`` the noise strength. The subscript ``Q`` refers to the
generalized norm ``||a||_Q^2 := \\langle a, Q^{-1} b \\rangle`` (see [`anorm`](@ref)). Here
``Q`` is the noise covariance matrix `sys.Σ`.

## Keyword arguments
* `cov_inv = nothing`: Inverse of the covariance matrix ``\\Sigma``.
  If `nothing`, ``\\Sigma^{-1}`` is computed automatically.
"""
function om_action(sys::StochSystem, path, time; cov_inv=nothing)
    # Inverse of covariance matrix
    (cov_inv == nothing) ? A = inv(sys.Σ) : A = cov_inv
    
    # Compute action integral
    S = 0
    for i in 1:(size(path, 2) - 1)
        S += sys.σ^2/2 * ((div_b(sys, path[:,i+1]) + div_b(sys, path[:,i]))/2 *
            (time[i+1]-time[i]))
    end
    fw_action(sys, path, time, cov_inv=A) + S/2
end;

"""
    action(sys::StochSystem, path::Matrix, time, functional; kwargs...)
Computes the action functional specified by `functional` for a given StochSystem `sys` and
`path` parameterized by `time`.

* `functional = "FW"`: Returns the Freidlin-Wentzell action ([`fw_action`](@ref))
* `functional = "OM"`: Returns the Onsager-Machlup action ([`om_action`](@ref))
"""
function action(sys::StochSystem, path::Matrix, time, functional; kwargs...)
    if functional == "FW"
        if is_multiplicative(sys)
            return fw_action(sys, path, time; cov_inv = cov_inv_along_path(sys, path, sys.pg, time))
        else
            return fw_action(sys, path, time; kwargs...)
        end
    elseif functional == "OM"
        return om_action(sys, path, time; kwargs...)
    end
end;

function geometric_action(sys::StochSystem, path::Matrix, arclength=1.0)
    if is_multiplicative(sys)
        A = cov_inv_along_path(sys, path, sys.pg, 0.)
    else # i.e. we do not have a multiplicative noise function
        A = inv(sys.Σ)
    end
    # calling the integrand values
    integrand = geom_fw_integrand(sys, path, arclength, A);
    # approximating the integral
    N = size(path,2); S = 0.;
    for ii in 1:N-1
        S += (integrand[ii] + integrand[ii+1])/2
    end
    S * arclength/(N-1)
end

"""
    geometric_action(sys::StochSystem, path, arclength=1; kwargs...)
Calculates the geometric action of a given `path` with specified `arclength` for the drift
field `sys.f`.

For a given path ``\\varphi``, the geometric action ``\\bar S`` corresponds to the minimum
of the Freidlin-Wentzell action ``S_T(\\phi)`` over all travel times ``T>0``, where ``\\phi``
denotes the path's parameterization in physical time (see [`fw_action`](@ref)). It is given
by the integral

``\\bar S[\\varphi] = \\int_0^L \\left( ||\\varphi'||_Q \\, ||b(\\varphi)||_Q - \\langle \\varphi', \\,
    b(\\varphi) \\rangle_Q \\right) \\, ds``

where ``s`` is the arclength coordinate, ``L`` the arclength, ``b`` the drift field, and the
subscript ``Q`` refers to the generalized dot product ``\\langle a, b \\rangle_Q := a^{\\top}
\\cdot Q^{-1} b`` (see [`anorm`](@ref)). Here ``Q`` is the noise covariance matrix `sys.Σ`.

## Keyword arguments
* `cov_inv = nothing`: Inverse of the covariance matrix ``\\Sigma``.
  If `nothing`, ``\\Sigma^{-1}`` is computed automatically. 

Returns the value of the geometric action ``\\bar S``.
"""
# function geometric_action(sys::StochSystem, path, arclength=1.0; cov_inv=nothing)
#     # defining the covariance matrix (across the path)
#     (cov_inv == nothing) ? A = inv(sys.Σ) : A = cov_inv
#     # calling the integrand values
#     integrand = geom_fw_integrand(sys, path, arclength, A)
#     # approximating the integral
#     S = 0
#     for ii in 1:N-1
#         S += (integrand[ii] + integrand[ii+1])/2
#     end
#     S * arclength/(N-1)
# end

"""
    fw_integrand(sys::StochSystem, path, time, A::Matrix)
Computes the squared ``A``-norm ``|| \\dot \\phi_t - b ||^2_A`` (see `fw_action` for
details). Returns a vector of length `N` containing the values of the above squared norm for
each time point in the vector `time`.

This is for the case where the covariance matrix `A` is fixed in time. See multiple dispatch method where the covariance matrix `A` is time-dependent: `fw_integrand(sys::StochSystem, path, time, A::Vector)`.
"""
function fw_integrand(sys::StochSystem, path, time, A::Matrix)
    v = path_velocity(path, time, order=4)
    sqnorm = zeros(size(path, 2))
    for i in 1:size(path, 2)
        drift = sys.f(path[:,i], p(sys), time[i])
        sqnorm[i] = anorm(v[:,i] - drift, A, square=true)
    end
    sqnorm
end;

"""
    fw_integrand(sys::StochSystem, path, time, A::Vector)
Computes the squared ``A_t``-norm ``|| \\dot \\phi_t - b ||^2_A_t`` (see `fw_action` for
details). Returns a vector of length `N` containing the values of the above squared norm for
each tim point ``t`` in the vector `time`.

This is for the case where the covariance matrix `A` is time-dependent. See multiple dispatch method where the covariance matrix `A` is fixed in tine: `fw_integrand(sys::StochSystem, path, time, A::Matrix)`.
"""
function fw_integrand(sys::StochSystem, path, time, A::Vector)
    v = path_velocity(path, time, order=4)
    sqnorm = zeros(size(path, 2))
    for i in 1:size(path, 2)
        drift = sys.f(path[:,i], p(sys), time[i])
        sqnorm[i] = anorm(v[:,i] - drift, A[i], square=true)
    end
    sqnorm
end;

"""
    geom_fw_integrand(sys, path, arclength, A::Matrix)
Computes the quantity ``|| \\varphi'_s ||_A|| \\boldsymbol{b}(\\varphi_s) ||_A - \\langle\\varphi'_s,\\boldsymbol{b}(\\varphi_s)\\rangle_A  `` (see `geometric_action` for
details). Returns a vector of length `N` containing the values of the above norm for
each point in the arclength-discretisation of the path.

This is for the case where the covariance matrix `A` is fixed in time. See the multiple dispatch method where the covariance matrix `A` is time-dependent: `geom_fw_integrand(sys::StochSystem, path, time, A::Vector)`.
"""
function geom_fw_integrand(sys, path, arclength, A::Matrix)
    N = size(path,2); # number of points in path space
    v = path_velocity(path, range(0, arclength, length=N), order=4); # the derivative of the path with respect to arc length
    drift = [sys.f(path[:,ii], p(sys), 0.) for ii ∈ 1:N]; # the drift field evaluated at each point
    integrand = [anorm(v[:,ii], A)*anorm(drift[ii], A) - dot(v[:,ii], A, drift[ii]) for ii ∈ 1:N];
    return integrand
end

"""
    geom_fw_integrand(sys, path, arclength, A::Vector)
Computes the quantity ``|| \\varphi'_s ||_A_s|| \\boldsymbol{b}(\\varphi_s) ||_A_s - \\langle\\varphi'_s,\\boldsymbol{b}(\\varphi_s)\\rangle_A_s  `` (see `geometric_action` for
details). Returns a vector of length `N` containing the values of the above norm for
each point in the arclength-discretisation of the path.

This is for the case where the covariance matrix `A` is time-dependent. See the multiple dispatch method where the covariance matrix `A` is fixed in tine: `geom_fw_integrand(sys::StochSystem, path, time, A::Matrix)`.
"""
function geom_fw_integrand(sys, path, arclength, A::Vector)
    N = size(path,2); # number of points in path space
    v = path_velocity(path, range(0, arclength, length=N), order=4); # the derivative of the path with respect to arc length
    drift = [sys.f(path[:,ii], p(sys), 0.) for ii ∈ 1:N]; # the drift field evaluated at each point
    integrand = [anorm(v[:,ii], A[ii])*anorm(drift[ii], A[ii]) - dot(v[:,ii], A[ii], drift[ii]) for ii ∈ 1:N];
    return integrand
end


"""
    div_b(sys::StochSystem, x)
Computes the divergence of the drift field `sys.f` at the given point `x`.
"""
function div_b(sys::StochSystem, x)
    b(x) = drift(sys, x)
    tr(ForwardDiff.jacobian(b, x))
end;

"""
    path_velocity(path, time; order=4)
Returns the velocity along a given `path` with time points given by `time`.

## Keyword arguments
* `order = 4`: Accuracy of the finite difference approximation.
  `4`th order corresponds to a five-point stencil, `2`nd order to a three-point stencil.
  In both cases, central differences are used except at the end points, where a forward or
  backward difference is used.
"""
function path_velocity(path, time; order=4)
    v = zeros(size(path))

    if order == 2
        # 1st order forward/backward differences for end points
        v[:,1] .= (path[:,2] .- path[:,1])/(time[2] - time[1])
        v[:,end] .= (path[:,end] .- path[:,end-1])/(time[end] - time[end-1])
        # 2nd order central differences for internal points
        for i in 2:(size(path, 2) - 1)
            v[:,i] .= (path[:,i+1] .- path[:,i-1])/(time[i+1] - time[i-1])
        end

    elseif order == 4
        # 1st order forward/backward differences for end points
        v[:,1] .= (path[:,2] .- path[:,1])/(time[2] - time[1])
        v[:,end] .= (path[:,end] .- path[:,end-1])/(time[end] - time[end-1])
        # 2nd order central differences for neighbors of end points
        v[:,2] .= (path[:,3] .- path[:,1])/(time[3] - time[1])
        v[:,end-1] .= (path[:,end] .- path[:,end-2])/(time[end] - time[end-2])
        # 4th order central differences for internal points
        for i in 3:(size(path, 2) - 2)
            v[:,i] .= ((-path[:,i+2] .+ 8*path[:,i+1] .- 8*path[:,i-1] .+ path[:,i-2])/
                (6*(time[i+1] - time[i-1])))
        end
    end
    v
end;
