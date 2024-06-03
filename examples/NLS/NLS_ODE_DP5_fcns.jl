struct NLS_ODE_cache{plantype,iplantype}
    omega::Float64
    U::Vector{ComplexF64}
    cU::Vector{ComplexF64}
    absU2::Vector{ComplexF64}
    FFT!::plantype
    IFFT!::iplantype
end

function expA!(u::Vector{ComplexF64},theta::Float64,p::NLS_ODE_cache{plantype,iplantype}) where {plantype,iplantype} 
    # u ->  exp(-i θ D^2) * u
    # where F * u = fft(u) and D  is the diagonal matrix with entries
    # (0, 1, 2, ... J2/2 -1, J2/2, J2/2-1, J2/2-2, ... 2,1)
    p.FFT!*u
    J2 = length(u)
    J = div(J2,2)
    aux = -im*theta
    u[J+1] *= exp(J*J*aux)
    for k in 1:J-1   # k = 0, 1, 2, ... J-1
            # λ = -im * k^2
            # e^{λθ }  or the reverse flow: e^{-λθ }
        expa = exp(k*k*aux)
        u[k+1] *=  expa  
        u[J2-k+1] *= expa
    end
    p.IFFT!*u
    return nothing
end

function expA(w::Vector{ComplexF64},theta::Float64,
              p::NLS_ODE_cache{plantype,iplantype}) where {plantype,iplantype} 
    u = copy(w)
    expA!(u,theta,p)
    return u
end

function g!(G,U,p)
    @. p.cU = conj(U)
    @. p.absU2 = U * p.cU
    @. G = im * p.absU2 * U
    return nothing
end


function NLS_ODE!(F, W, p, t)
    U = p.U
    theta = p.omega*t
    U .= W
    expA!(U,theta,p)  
    g!(F,U,p)
    expA!(F,-theta,p)
    return nothing
end

function NLS_ODE_cache_init(u0::Vector{ComplexF64})
    U = similar(u0) 
    cU = similar(u0) 
    absU2 = similar(u0)
    FFT! = plan_fft!(u0)
    IFFT! = inv(FFT!)
    return NLS_ODE_cache(omega,U,cU,absU2,FFT!,IFFT!)
end
