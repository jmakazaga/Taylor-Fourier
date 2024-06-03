struct J2_VOP_ODE_TF_cache{ueltype}  
    parms::Vector{Float64}
    c_vec::Vector{Float64}
    c_vec_::Vector{Float64}
    sdiv_vec::Vector{Float64}
    Uaux::TaylorFourierSeries{ueltype,Float64}
end

function TF_ODE!(FU, U, cache) 
    # This function makes the computations indicated in (7) in the article. 
    # Here, U = \sum_{j=0}^{d} t^j Y_{n,j}, and on output, FU = \sum_{j=0}^{d} t^j Z_{n,j}
    cvec= cache.c_vec  # c = cos(θ), precomputed for each $\theta = \theta_j$ in $[0, 2π]$
    cvec_= cache.c_vec_  # c = -cos(θ), precomputed for each $\theta = \theta_j$ in $[0, 2π]$
    svec= cache.sdiv_vec # s = sin(θ)/ω precomputed for each $\theta = \theta_j$ in $[0, 2π]$
    C = cache.parms[2]
    # α = u[1:4]
    # β = u[5:8]
    # ulag = c * α + s * β
    Uaux = cache.Uaux
    #ulag = c * u[1:4] + s * u[5:8]
    mult!(Uaux,1,U,1,cvec)
    mult!(Uaux,2,U,2,cvec)
    mult!(Uaux,3,U,3,cvec)
    mult!(Uaux,4,U,4,cvec)
    mult!(Uaux,5,U,5,svec)
    mult!(Uaux,6,U,6,svec)
    mult!(Uaux,7,U,7,svec)
    mult!(Uaux,8,U,8,svec)
    add!(Uaux,1,Uaux,5) #Uaux[1] = c * u[1] + s * u[5]
    add!(Uaux,2,Uaux,6) #Uaux[2] = c * u[2] + s * u[6]
    add!(Uaux,3,Uaux,7) #Uaux[3] = c * u[3] + s * u[7]
    add!(Uaux,4,Uaux,8) #Uaux[4] = c * u[4] + s * u[8]
            # So ulag = Uaux[1:4]
    mult!(Uaux,5, Uaux,1, Uaux,3)
    mult!(Uaux,6, Uaux,2, Uaux,4)
    add!(Uaux,5, Uaux,6)
    mult!(Uaux,5,2.0)    #           Uaux[5] = z = 2*(u[1]*u[3] + u[2]*u[4])
    
    mult!(Uaux,6, Uaux,1, Uaux,1)
    mult!(Uaux,7, Uaux,2, Uaux,2)
    add!(Uaux,6, Uaux,7) # uaux[6]= u[1]^2 + u[2]^2
    mult!(Uaux,7, Uaux,3, Uaux,3)
    add!(Uaux,6, Uaux,7) # uaux[6]= u[1]^2 + u[2]^2 + u[3]^2 
    mult!(Uaux,7, Uaux,4, Uaux,4)
    add!(Uaux,6, Uaux,7) #            uaux[6]= r = u[1]^2 + u[2]^2 + u[3]^2 + u[4]^2
 
    
    power!(Uaux,7,Uaux,6,-3)  #       uaux[7]= w = 1/r^3

    
    mult!(Uaux,9, Uaux,6, Uaux,6) # uaux[8]= r^2
    mult!(Uaux,8, Uaux,9, Uaux,7)  # uaux[8]= 1/r = r^2/r^3     
    mult!(Uaux,9, Uaux,5, Uaux,8)   #   uaux[9]= sinth = z/r  
    
    mult!(Uaux,6, Uaux,9, Uaux,9) #   uaux[6] = sinth^2
    
    mult!(Uaux,5, Uaux,7, Uaux,9) # uaux[5] = w*sinth
    mult!(Uaux, 5, 1.5*C)   #         uaux[5] = B = 1.5*C*w*sinth
    
    mult!(Uaux,9, Uaux,7, Uaux,6)  # uaux[9] = w*sinth^2
    mult!(Uaux,9, -3.0*C)  # uaux[9] = 0.5*C*w*(-6)*sinth^2
    mult!(Uaux,7, 0.5*C)  # uaux[7] = 0.5*C*w
    add!(Uaux,6, Uaux,9, Uaux,7)  #           Uaux[6] = A = 0.5*C*w*(1 - 6*sinth^2)
    
    #return into FU[1:4]
    mult!(FU, 1, Uaux, 6, Uaux, 1)  # A*u[1]
    mult!(FU, 5, Uaux, 5, Uaux, 3)  # B*u[3]
    add!(FU, 1, FU, 5)  #            FU[1] = A*u[1]+B*u[3]
    
    mult!(FU, 2, Uaux, 6, Uaux, 2)  # A*u[2]
    mult!(FU, 5, Uaux, 5, Uaux, 4)  # B*u[4]
    add!(FU, 2, FU, 5)  #             FU[2] = A*u[2]+B*u[4]
    
    mult!(FU, 3, Uaux, 6, Uaux, 3) # A*u[3]
    mult!(FU, 5, Uaux, 5, Uaux, 1) # B*u[1]
    add!(FU, 3, FU, 5)  #             FU[3] = A*u[3]+B*u[1]
    
    mult!(FU, 4, Uaux, 6, Uaux, 4)  # A*u[4]
    mult!(FU, 5, Uaux, 5, Uaux, 2)  # B*u[2]
    add!(FU, 4, FU, 5)  #             FU[4] = A*u[4]+B*u[2]
    #  END of gradR(Uaux)
    
    # Now gradR(Uaux) is stored in FU[1:4]:   aux = gradR(Uaux) = FU[1:4]

    
    # dα = s*aux
    # dβ = -c*aux
    # du[5:8].= -c*aux   

    
    # println("cos = ", cvec)
    mult!(FU, 5, FU, 1, cvec_)
    mult!(FU, 6, FU, 2, cvec_)
    mult!(FU, 7, FU, 3, cvec_)
    mult!(FU, 8, FU, 4, cvec_)
    # du[1:4].= s*aux
    # println("sin = ", svec)   
    mult!(FU, 1, svec)
    mult!(FU, 2, svec)
    mult!(FU, 3, svec)
    mult!(FU, 4, svec)
    # dt = ||ulag||^2
    # du[9]=dot(ulag,ulag) #ulag[1]*ulag[1]+ulag[2]*ulag[2]+ulag[3]*ulag[3]+ulag[4]*ulag[4]
    mult!(FU, 9, Uaux, 1, Uaux, 1)  # FU[9] = ulag[1]*ulag[1]
    mult!(Uaux, 1, Uaux, 2, Uaux, 2)  # ulag[1] = ulag[2]*ulag[2]
    add!(FU,9, Uaux, 1)              # FU[9] = ulag[1]*ulag[1] + ulag[2]*ulag[2]
    mult!(Uaux, 1, Uaux, 3, Uaux, 3) # ulag[1] = ulag[3]*ulag[3]
    add!(FU,9, Uaux, 1)              # FU[9] = ulag[1]*ulag[1] + ulag[2]*ulag[2] + ulag[3]*ulag[3]
    mult!(Uaux, 1, Uaux, 4, Uaux, 4)  # ulag[1] = ulag[4]*ulag[4]
    add!(FU,9, Uaux, 1)   # FU[9] = ulag[1]*ulag[1] + ulag[2]*ulag[2] + ulag[3]*ulag[3] +  ulag[4]*ulag[4]
    return nothing
end

function J2_VOP_cache_init(U0,parms,ω,deg,M)
    thetas = range(0.,step=π/M, length=2M)
    c_vec = cos.(thetas)
    c_vec_ = -c_vec
    s_vec = sin.(thetas)/ω
    TFaux=TaylorFourierSeries(U0,ω,deg,M)
    cache = J2_VOP_ODE_TF_cache(parms,c_vec,c_vec_,s_vec,TFaux)
    return cache
end
