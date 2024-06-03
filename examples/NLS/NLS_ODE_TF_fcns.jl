struct NLS_ODE_TF_cache{plantype,iplantype}
    G::TaylorFourierSeries{ComplexF64,Float64}
    cU::TaylorFourierSeries{ComplexF64,Float64}
    absU2::TaylorFourierSeries{ComplexF64,Float64}
    FFT!::plantype
    IFFT!::iplantype
    exp_array::Array{ComplexF64,2}
    exp_array_::Array{ComplexF64,2}
end


function expA!(W::TaylorFourierSeries{ComplexF64,Float64},
               reverse::Bool,
               p::NLS_ODE_TF_cache{plantype,iplantype}) where {plantype,iplantype}

    FFT! = p.FFT!
    IFFT! = p.IFFT!
    U = W.tcoeffs
    FFT!*U
    M = W.M                           # number of theta points in [0,2pi] is 2M
    num_coefs = W.working_degree[1]+1 # degree of taylor series + 1 (d+1)
    J2 = W.num_vars                   # number of variables (2J)
    exp_array = p.exp_array
    if reverse
       exp_array = p.exp_array_
    end
    @inbounds for g in 1:num_coefs
        for j in 1:J2
            for k in 1:2M
                U[k,j,g] *= exp_array[k,j]
            end
        end
    end
    IFFT!*U
    return nothing
end

   
function g!(G::TaylorFourierSeries{ComplexF64,Float64},
            U::TaylorFourierSeries{ComplexF64,Float64},
            p::NLS_ODE_TF_cache{plantype,iplantype}) where {plantype,iplantype}
    conj!(p.cU,U)
    mult!(p.absU2,U,p.cU)
    mult!(G,p.absU2,U)
    mult!(G,im)
    G.working_degree[1]=U.working_degree[1]
    return nothing
end


function NLS_ODE_TF!(FW::TaylorFourierSeries{ComplexF64,Float64},
            W::TaylorFourierSeries{ComplexF64,Float64},
            p::NLS_ODE_TF_cache{plantype,iplantype}) where {plantype,iplantype}
    # Evaluates f(theta,W) for theta=(j-1)*pi/M,  j=1,...,2M.
    expA!(W,false,p) 
    g!(FW,W,p)
    expA!(FW,true,p)
    return nothing
end

function NLS_ODE_TF_cache_init(W0::Vector{ComplexF64},omega,deg,M)
    J2 = length(W0)
    J = div(J2,2)
    M2 = 2M
    G=TaylorFourierSeries(W0,omega,deg,M)
    cU=TaylorFourierSeries(W0,omega,deg,M)
    absU2=TaylorFourierSeries(W0,omega,deg,M)
    FFT! = plan_fft!(G.tcoeffs,2)
    IFFT! = plan_ifft!(G.tcoeffs,2)
    exp_array = Array{ComplexF64}(undef,M2,J2)
    exp_array_ = Array{ComplexF64}(undef,M2,J2)

    dtheta = pi/M
    @inbounds for k in 1:M2    
        exp_array[k,1] = 1.0 
        exp_array_[k,1] = 1.0
        theta= (k-1)*dtheta   # θ_k angle in [0,pi]  
        aux = im*theta
        jaux = J*J*aux
        exp_array[k,J+1] = exp(-jaux)
        exp_array_[k,J+1] = exp(jaux)
        for j in 1:J-1
            jaux = j*j*aux
            exp_array[k,j+1] = exp(-jaux)
            exp_array_[k,j+1] = exp(jaux)
            ind = J2-j+1
            exp_array[k,ind] = exp(-jaux)
            exp_array_[k,ind] = exp(jaux)
        end
    end
    return NLS_ODE_TF_cache(G,cU,absU2,FFT!,IFFT!,exp_array,exp_array_)
end
