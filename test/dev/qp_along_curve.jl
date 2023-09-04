function qp_along_curve(sys::StochSystem, curve::Matrix, ref_pnt::Vector)
    
    qp_along_curve = zeros(size(curve,2)); 
    
    @Threads.threads for ii ∈ 1:size(curve,2)
        qp_along_curve[ii] = minimum(geometric_min_action_method(sys, ref_pnt, curve[:,ii]; maxiter = 1e3, converge = 1e-8, showprogress = false)[2])  
    end

    qp_along_curve

end