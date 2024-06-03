
using FFTW
using LinearAlgebra

#=
This file contains a structure, TaylorFourierSeries, which is used to solve a periodic differential equation with either Float64 values or ComplexF64 values.
The modulated serie of a variable is of the form:


ms(t,\theta) = sum_{g=0}^max_degree t^g  Sum_{k=-M}^M v_{k,g} exp(i k \theta) +
               +t^(max_degree+1) p_(max_degree+1)

(note1: for negative k indices we use 2M+k with k<0)
(note2: for k=M and k=-M we have v_{M,g} = v_{-M,g})
(note3: \theta is the angular variable and it is used to distribute 2M points in the period: \theta = 2*pi/period*t)
(note4: the coefficients of each degree of t is a trigonometric polinomial of degree M in \theta,
        but the highest degree of t has a constant coefficient)

Taking into account the notes we can  write it as (1):

sum_{g=0}^max_degree t^g (v_{0,g} + Sum_{k=1}^{M-1} v_{k,g} exp(i k \theta) + 1/2 v_{M,g} exp(i M \theta) +  
                                    Sum_{k=1}^{M-1} v_{2M-k,g} exp(-i k \theta) + 1/2 v_{M,g} exp(-i M \theta) +
                     t^(max_degree+1) v_(0,max_degree+1)

where each coefficient of t is a trigonometric interpolator (fourier serie), and for the higher order it is a constant value.
So the modulated-serie needs a max_order and m value.

The modulated-series interpolate the values of regularly distributed 2M points over the period (2M different 
values of \theta distributed on the period). 
For these values we have the Taylor polinomials of the given degree: we have 2M points in the period and for each 
point we have a Taylor polynomial of the given degree for each state variable. 
If we look to the coefficients of a given degree in the Taylor polynomials (there are 2M polinomials, so 2M coefficients) we can interpolate them with a trigonometric polinomial (fourier series): that is, each trigonometric interpolator (the Fourier series) interpolates the 2M coefficients of a given degree of the 2M points of the period. 
So we can recover the Taylor polynomials from the modulated series (evaluating it for a given fixet \theta): ms(t,k) returns Taylor1(t) at ŧheta = k

In the structure we have the coefficients of the modulated-series (1) in the matrix coeffs, but we have also the Taylor series of
the 2*M points over the period, they are in tcoeffs matrix.

To build the modulated-series we need the problem dependant function that given the modulated-series returns another modulated-series with the new Taylor series. 
The generation of the modulated-series is a process similar to the process used in the Taylor method:
Starting with a basic modulated-series (of a given order) we evaluate the function of the ODE on the modulated-series, as result we get another modulated-series and  we "integrate" the modulated-series, so
we get a modulated series of superior order (the given order + 1). 
The evaluation of the function on the modulated-series requires some operations on the structure, so this file offers basic operations for TaylorFourierSeries that act over the tcoeffs matrix. (the operations are for Taylor polinomials)
The basic operations over TaylorFourierSeries defined are:
  addition:
      msa[i] = msb[j] + msc[k]
      msa[i] = msa[i] + msb[j]   
  multiplication: 
      msa = masb * escalar 
      msa = msa * escalar
      msa[i] = msb[j] * msc[k]
      msa[i] = msa[i] * msb[j] 
      msa[i] = msb[j]* vector_of_scalars   // one different scalar for each 2M points
  power:
      msa[i] = (msb[j])^exp
  conjugate:
      msa = conj(msb)
      msa[i] = conj(msb[j])

We offer the function to generate the problem:

problem= Real_PeriodicODEProblem(funct!, params, u0, omega)

where:
.- funct! is the function of the ODE and it is called as
       funct!(msb,msa,params)
   so
       msb = funct(msa)
   and params is a user defined structure in which the user can pass the parametters needed by funct!
.- params is a user deffined structure with the information needed in the evaluation of funct!
.- u0 is the initial value of the variables (a vector of initial values)
.- omega is 2*pi/period.

once we have defined the problem we can solve it with

ms = solve(problem, max_degree, m)

where:
.- problem is the problem defined by a call to Real_PeriodicODEProblem
.- max_degree is the Taylor series degree
.- m is the trigonometric polinomial degree
It returns the Real_TaylorFourierSeries structure with the modulated-series expansion and the Taylor series expansions on the 2M points.
The Real_TaylorFourierSeries can be evaluated at any point with the function:

evaluateTaylorFourierSeries(ms::Real_TaylorFourierSeries,t)  #y(t)--> y(t,omega*t) 

where ms is the Real_TaylorFourierSeries to be evaluated, t is the time at witch the modulated serie will be 
evaluated (ms(t,omega*t)) (omega = 2*pi/P is stored in the problem, which is storen in the Real_TaylorFourierSeries structure returned by solve)

=#

struct PeriodicODEProblem{uelType,tType,pType}
  funct!::Function
  params::pType
  u0::Vector{uelType}
  omega::tType
end


struct TaylorFourierSeries{uelType,tType}
  working_degree::Array{Int64,0}  # The degree reached when generating the modulated serie. At the begining it is 0.
  omega::tType # 2*pi/period
  num_vars::Int64   # number of variables of the problem
  max_degree::Int64 # maximun degree of the polynomials. So each polynomial has max_degree+1 coefficients
  M::Int64 # M value of the Fourier serie --> 2M points in the period...
  coeffs::Array{ComplexF64,3} # 3D matrix to store the modulated series. 
        # We have 1 "modulated serie" for each variable (num_vars variables) 
        # We have 1 "Fourier serie" for each coefficient of the polynomial (max_degree+1 coefficients) 
        # We have 1 value of type ComplexF64 for each coefficient of the Fourier serie (M+1 coefficients)
        # So, we need (M+1) * (max_degree+1) * num_vars elements.
        # The modulated serie for each variable is: 

        # sum_{g=1}^max_degree t^(g-1) (v[1,g} + Sum_{k=2}^{M} 2 * v_{k,g} exp(i*(k-1) \thetha) +  v_{M+1,g} exp(i M \theta) 
        #                    + t^(max_degree) v_(1,working_degree+1)
             
        # note that:
        # We are working only with the half of the trigonometric polinomial: there are not exp(-i*k*|theta) Because for real
        #          values they are conjugate values...
        # the last Fourier Serie has only the constant coefficient 
        # the v_{k,g} values are NOT the values returned by fft. fft returns v_{k,g}*2*M values (we should have to divide 
        #    the evaluation of the expresion above if we had the values returned by fft)
  tcoeffs::Array{uelType,3}  # 3D matrix to store the Taylor Series coefficients
        # We have 2M points distributed in the period, and we have to evaluate the function in all of them.
        # To evaluate the function we need the Taylor series of the value of each variable in each point.
        # and each Taylor Serie is a polynomial of degree max_degree (they have max_degree+1 coefficients)
        # So we need  2*M x (max_degree+1) x num_vars coefficients.
        # The evaluation of the function will return a matrix of the same form.
end 



function TaylorFourierSeries(u0::Vector{Float64}, omega::Float64, max_degree::Int64, m::Int64)
    num_vars = length(u0)
    ms = TaylorFourierSeries{Float64,Float64}(Array{Int64,0}(undef), omega, length(u0), max_degree, m, Array{ComplexF64}(undef,m+1, num_vars,max_degree+1), zeros(Float64,2m, num_vars,max_degree+1))
    return ms
end

function TaylorFourierSeries(u0::Vector{ComplexF64}, omega::Float64, max_degree::Int64, m::Int64)
    num_vars = length(u0)
    ms = TaylorFourierSeries{ComplexF64,Float64}(Array{Int64,0}(undef), omega, length(u0), max_degree, m, Array{ComplexF64}(undef,2m, num_vars,max_degree+1), zeros(ComplexF64,2m, num_vars,max_degree+1))
    return ms
end



struct PeriodicODEProblemSolution{uelType,tType,pType}
  t::Vector{tType}  
  u::Vector{Vector{uelType}}  
  TFS::TaylorFourierSeries{uelType,tType}
  prob::PeriodicODEProblem{uelType,tType,pType}
end




function TaylorFourierSolve(prob::PeriodicODEProblem{ueltype,Float64,pType}, max_degree::Int64,m::Int64, n_orbits=0) where {ueltype,pType}
    u0 = prob.u0
    msa=TaylorFourierSeries(u0, prob.omega, max_degree, m)
    msb=TaylorFourierSeries(u0, prob.omega, max_degree, m)
    TaylorFourierSeries_init!(msa,prob) # gets initial Taylor-series
    for w_degree in 1:max_degree 
       prob.funct!(msb,msa,prob.params)  # evaluate the Differential Equation at each point in the period
                                          # generate the Taylor series of f(msa)
       msb.working_degree[] = msa.working_degree[]
       interpolate_T_coeffs_by_Fourier!(msb) # get the interpolating Fourier series for all 
                                                       # coefficients of each degree of the Taylor series
       integrate!(msa, msb, u0) # integrate and increase order 
       mod_serie2taylor_series!(msa)  # Get the Taylor-series from the new modulated-series
    end
    tt, uu = ComputeDiscreteSolution(msa,n_orbits)
    return PeriodicODEProblemSolution(tt,uu,msa,prob)
end
 

function (sol::PeriodicODEProblemSolution{uelType,tType,pType})(t::Number) where {uelType,tType,pType}
    return sol.TFS(t)
end
    

function TaylorFourierSeries_init!(msa::TaylorFourierSeries{uelType,tType},prob::PeriodicODEProblem{uelType,tType,pType}) where {uelType,tType,pType}
    # initialization of the Taylor series
    TwoM = 2*msa.M
    @inbounds for  i in 1:length(prob.u0)
        u0i = prob.u0[i]
        for j in 1:TwoM
            msa.tcoeffs[j,i,1]=u0i
        end
    end
    msa.working_degree[] = 0
    return nothing
end



function interpolate_T_coeffs_by_Fourier!(msa::TaylorFourierSeries{Float64,Float64})
    # The first dimension of the matrix tcoeffs has the values to be interpolated
    # get the fourier series that interpolate the Taylor coefficients
    ind = 1:msa.working_degree[]+1
    TS = view(msa.tcoeffs,:,:,ind)
    MS = view(msa.coeffs,:,:,ind)
#    MS .= rfft(TS,1)  # This is equivalent to the two lines below
    plan = plan_rfft(TS,1)
    MS .= plan * TS
    return nothing
end

function interpolate_T_coeffs_by_Fourier!(msa::TaylorFourierSeries{ComplexF64,Float64})
    # The first dimension of the matrix tcoeffs has the values to be interpolated
    # get the fourier series that interpolate the Taylor coefficients
    ind = 1:msa.working_degree[]+1
    TS = view(msa.tcoeffs,:,:,ind)
    MS = view(msa.coeffs,:,:,ind)
#    MS .= fft(TS,1)  # This is equivalent to the two lines below
    plan = plan_fft(TS,1)
    MS .= plan * TS
    return nothing
end


# mod_serie2taylor_series! Obtains the Taylor Series from the modulated serie.
function mod_serie2taylor_series!(msa::TaylorFourierSeries{Float64,Float64})
    M = msa.M
    TwoM=2*M
    number_of_Taylor_coefs = msa.working_degree[]+1
    ind = 1:number_of_Taylor_coefs
    MS = view(msa.coeffs,:,:,ind)
    TS = view(msa.tcoeffs,:,:,ind)
#    TS .= irfft(MS,TwoM,1) # This is equivalent to the two lines below
    plan = plan_irfft(MS,TwoM,1)
    TS .= plan * MS
    # now we have the modulated series and taylor series
    return nothing
end


# mod_serie2taylor_series! Obtains the Taylor Series from the modulated serie.
function mod_serie2taylor_series!(msa::TaylorFourierSeries{ComplexF64,Float64})
    M = msa.M
    TwoM=2*M
    number_of_Taylor_coefs = msa.working_degree[]+1
    ind = 1:number_of_Taylor_coefs
    MS = view(msa.coeffs,:,:,ind)
    TS = view(msa.tcoeffs,:,:,ind)
#    TS .= ifft(MS,1) # This is equivalent to the two lines below
    plan = plan_ifft(MS,1)
    TS .= plan * MS
    return nothing
end




function integrate!(msa::TaylorFourierSeries{Float64,Float64}, z::TaylorFourierSeries{Float64,Float64}, u0::Vector{Float64})
    # The process start with the coefficient of the "working_degree" degree (the highest degree) wich is constant
    #           without sinusoidal elements.
    omega = msa.omega  # 2.0*pi/P
    m =msa.M
    TwoM = 2*m
    number_of_vars=msa.num_vars
    number_of_Taylor_coefs=z.working_degree[]+1
    
    # Initially the values of \hat{Y}_{d+1} = 0 , for the working_degree+1  are 0 
    @inbounds for elem in 1:number_of_vars
        # The highest degree only has the constant coefficient (no sinusoidal coefficients are present).
        # it is stored as the first coefficient
        # The constant value (for each degree) is assigned at the begining of the body-code of the for:
        # before the coefficients of the sinusoidal polinomial we assign the constant coeff of the previous degree 
        # q_{i+1} = p_i / i 
        # @inbounds msa.coeffs[1,number_of_Taylor_coefs+1,elem]= z.coeffs[1,number_of_Taylor_coefs,elem]/number_of_Taylor_coefs
        # the rest of values are 0
        for k in 2:m+1   
            msa.coeffs[k,elem,number_of_Taylor_coefs+1]= 0.0
        end
    end
    # now, in reverse order, starting with d and going down: j in d:-1:0
    # y_{k,j} = (z_{k,j} − (j + 1)y_{k,j+1}) / ikω 
    @inbounds for g in number_of_Taylor_coefs:-1:1  # from higher order to lower order!!  
        for elem in 1:number_of_vars
            # First, the constant coefficient of the previous degree
            # (the higher degree only has this coefficient)
            # q_{i+1} = p_i / i
            msa.coeffs[1,elem,g+1]= z.coeffs[1,elem,g]/g
            # Second the trigonometric coefficients
            for k in 2:m   # for the sinusoidal coefficients
                # from 2 to (M-1) 
                msa.coeffs[k,elem,g]= (z.coeffs[k,elem,g]-g*msa.coeffs[k,elem,g+1])/(im*(k-1)*omega)
                # rfft works only with the half of the coefficients!
                # from 2M to (M+1)  (these are the indices that go from -(M-1) to -1), negative!
                #@inbounds msa.coeffs[2*m-k+2,g,elem]= -(z.coeffs[2*m-k+2,g,elem]-g*msa.coeffs[2*m-k+2,g+1,elem])/(im*(k-1)*omega)
            end
            # The last element is the sum of y_{M,g} + y_{-M,g}
            # but y_{M,max_order} = y_{-M,max_order} = z_{M,max_order}  so we would lost the values y_{M,g}
            # Take into acount that z_{M,g} are real values (complex part = 0),
            # we have to compute them separately...
            #  not this way: @inbounds msa.coeffs[m+1,g,elem]= (z.coeffs[m+1,g,elem]-g*msa.coeffs[m+1,g+1,elem])/(im*m*omega)
        end
    end
    # we will compute the M coefficient separatelly:
    # wi will avoid the use of temporal vectors, so the process has to be for each element
    # the y_{M,max_degree+1} coefficients are 0
    iMomega_factor = 1.0/(im*m*omega)
    @inbounds for elem in 1:number_of_vars
        # y_{M,max_order} + y_{-M,max_order} = 0 
        # this is because 
        # y_{M,max_order} = z_{M,max_order}/(im*M*omega) 
        # and 
        # y_{-M,max_order} = z_{-M,max_order}/(-im*M*omega)
        # but 
        # z_{M,max_order} = z_{-M,max_order} so the sum is 0
        msa.coeffs[m+1,elem,number_of_Taylor_coefs]=zero(ComplexF64)
        # we will need the previous y_{M,j} value! so we store it in an auxiliar var. Initially:
        yMprev = (z.coeffs[m+1,elem,number_of_Taylor_coefs]/2.0)*iMomega_factor
        # println("first yM and YminusM = ",yMprev,)
        for g in number_of_Taylor_coefs-1:-1:1  # from higher order to lower order!! 
            yMprev = ((z.coeffs[m+1,elem,g]/2.0)-g*yMprev)*iMomega_factor
            # y_{M,g} + y_{-M,g} =  y_{M,g} + conj(y_{M,g}) = 2*real(y_{M,g})
            # println("rest yM and YminusM = ",yMprev)
            msa.coeffs[m+1,elem,g]= 2.0*real(yMprev)
        end
    end
    #end of  M coeeficients
    # To get q_1 we have to solve u(0)=u_0 
    @inbounds  for elem in 1:number_of_vars
        xpart= msa.coeffs[m+1,elem,1]
        for k in m:-1:2
            xpart= xpart + 2.0 * msa.coeffs[k,elem,1]
        end
        msa.coeffs[1,elem,1] = TwoM*u0[elem] - xpart 
    end
    # we have increased the degree of the modulated serie
    msa.working_degree[] = z.working_degree[]+1
    return nothing
end
 

function integrate!(msa::TaylorFourierSeries{ComplexF64,Float64}, z::TaylorFourierSeries{ComplexF64,Float64}, u0::Vector{ComplexF64})
    # The process start with the coefficient of the "working_degree" degree (the highest degree) wich is constant
    #           without sinusoidal elements.
    omega = msa.omega  # 2.0*pi/P
    m =msa.M
    number_of_vars=msa.num_vars
    number_of_Taylor_coefs=z.working_degree[]+1
    #  WAY_1  y_m = Array{ComplexF64}(undef,2,number_of_Taylor_coefs+1,number_of_vars)
    
    # Initially the values of \hat{Y}_{d+1} = 0 for the working_degree+1 
    @inbounds for elem in 1:number_of_vars
        # The highest degree only has the constant coefficient (no sinusoidal coefficients are present).
        # it is stored as the first coefficient on coeffs matrix
        # The constant coefficient value is 
        # q_{i+1} = p_i / i 
        # @inbounds msa.coeffs[1,number_of_Taylor_coefs+1,elem]= z.coeffs[1,number_of_Taylor_coefs,elem]/number_of_Taylor_coefs
        # the rest of values are 0 for the highest degree
        for k in 2:2*m   
            msa.coeffs[k,elem,number_of_Taylor_coefs+1]= 0.0
        end
    end
    # now, in reverse order, starting with d and going down: j in d:-1:0
    # y_{k,j} = (z_{k,j} − (j + 1)y_{k,j+1}) / ikω
    @inbounds for g in number_of_Taylor_coefs:-1:1  # from highest order dowto!!  
        for elem in 1:number_of_vars
            # the constant coefficients depends only on z values
            # (the highest degree only has this coefficient)
            # q_{i+1} = p_i / i
            # and the lowest degree must be derived from the initial value
            msa.coeffs[1,elem,g+1]= z.coeffs[1,elem,g]/g
            for k in 2:m   # for the sinusoidal coefficients, all but M and -M
                # from 2 to (M-1) 
                msa.coeffs[k,elem,g]= (z.coeffs[k,elem,g]-g*msa.coeffs[k,elem,g+1])/(im*(k-1)*omega)
                # from 2M to (M+1)  (these are the indices that go from -(M-1) to -1), negative!
                msa.coeffs[2*m-k+2,elem,g]= -(z.coeffs[2*m-k+2,elem,g]-g*msa.coeffs[2*m-k+2,elem,g+1])/(im*(k-1)*omega)
            end
            # For the last element we only have one value \hat{Z}_{M,j} = z_{M,j} + z_{-M,j}
            # and z_{M,j} = z_{-M,j} so z_{M,j} = \hat{Z}_{M,j}/2
            # for y_{M,j} and for y_{-M,j}
            # the integration for the values y_{M,j} and y_{-M,j} gives for j= max_degree:
                # y_{M,j} = z_{M,j} / (i M \omega}
                # y_{-M,j} = -z_{M,j} / (i M \omega}
            # and for the rest of j values:
                # y_{M,j} = (z_{M,j} - d y_{M,j+1}) / (i M \omega} 
                # y_{-M,j} = (-z_{M,j} - d y_{-M,j+1}) / (i M \omega}
            #= WAY_1
                # so, we need the values y_{M,j} and y_{-M,j} separated...
            y_m[1,g,elem] = (z.coeffs[m+1,g,elem]/2.0 - g*y_m[1,g+1,elem])/(im*m*omega)
            #  y_{M,j}
            y_m[2,g,elem] = (-z.coeffs[m+1,g,elem]/2.0 + g*y_m[2,g+1,elem])/(im*m*omega)
            # And we save in the structure the sum of both: y_{M,j} + y_{-M,j}
            @inbounds msa.coeffs[m+1,g,elem]= y_m[1,g,elem] + y_m[2,g,elem] 
            =#
            
            # @inbounds msa.coeffs[m+1,g,elem]= (z.coeffs[m+1,g,elem]-g*msa.coeffs[m+1,g+1,elem])/(im*m*omega)
        end
    end
    # WAY_2 
    # now we are going to save y_{M,j} + y_{-M,j} into the coeffs matrix (position m+1)
    # we have as general recursion:
    # y_{M,j} = (z_{M,j} − (j + 1)y_{M,j+1}) / iMω 
    # and
    # y_{-M,j} = (-z_{-M,j} + (j + 1)y_{-M,j+1}) / iMω 
    
    # starting from the max_degree:
    # y_{M,max_degree} = z_{M,max_degree}/ iMω 
    # y_{-M,max_degree} = -z_{-M,max_degree}/ iMω 
    # but 
    # z_{M,max_degree} = z_{-M,max_degree} = z.coeffs[m+1,max_degree,elem]/2.0
    # so the sum y_{M,max_degree}  + y_{-M,max_degree} is 0:
    #  y_{M,max_degree}  + y_{-M,max_degree}  = z_{M,max_degree}/ iMω - z_{-M,max_degree}/ iMω =
    #                 =  (z.coeffs[m+1,max_degree,elem]/2.0 -  z.coeffs[m+1,max_degree,elem]/2.0) /iMω  = 0
    # so if we save the sum we will lose y_{M,max_degree} and  y_{-M,max_degree} values
    # To avoid the lost of these values, the solution is to store them into auxiliar variables, and in reverse order,
    # go computing the rest of values:
    # that is:
    # the sum  y_{M,j} + y_{-M,j} to be saved into the coeffs matrix
    # and the auxiliar values yMprev and yminusMprev to be used into the next degree.
    
    # we will compute the M coefficient separatelly:
    # we will avoid the use of temporal vectors, so the process has to be for each element
    # the y_{M,max_degree+1} coefficients are 0
    iMomega_factor = 1.0/(im*m*omega)
    @inbounds for elem in 1:number_of_vars
        # y_{M,max_order} + y_{-M,max_order} = 0 
        # this is because 
        # y_{M,max_order} = z_{M,max_order}/(im*M*omega) 
        # and 
        # y_{-M,max_order} = z_{-M,max_order}/(-im*M*omega)
        # but 
        # z_{M,max_order} = z_{-M,max_order} so the sum is 0
        msa.coeffs[m+1,elem,number_of_Taylor_coefs]=zero(ComplexF64)
        # we will need the previous y_{M,j} value! so we store it in an auxiliar var. Initially:
        yMprev = (z.coeffs[m+1,elem,number_of_Taylor_coefs]/2.0)*iMomega_factor
        yminusMprev = -(z.coeffs[m+1,elem,number_of_Taylor_coefs]/2.0)*iMomega_factor
        # println("first yM and YminusM = ",yMprev," and ",yminusMprev)
        for g in number_of_Taylor_coefs-1:-1:1  # from higher order to lower order!! 
            yMprev = ((z.coeffs[m+1,elem,g]/2.0)-g*yMprev)*iMomega_factor
            yminusMprev = -((z.coeffs[m+1,elem,g]/2.0)-g*yminusMprev)*iMomega_factor
            # println("rest yM and YminusM = ",yMprev," and ",yminusMprev)
            # y_{M,g} + y_{-M,g}
            msa.coeffs[m+1,elem,g]= yMprev + yminusMprev
        end
    end
    #end of  M coeeficients
    
    # To get q_1 we have to solve q_1(0)=u_0 
    @inbounds for elem in 1:number_of_vars
        xpart= msa.coeffs[m+1,elem,1]
        ypart = zero(ComplexF64)
        for k in m:-1:2
            xpart= (xpart + msa.coeffs[k,elem,1])
            ypart = (ypart+ msa.coeffs[2*m-k+2,elem,1])
        end
        msa.coeffs[1,elem,1] = u0[elem]*2*m - (xpart+ypart) 
    end
    # we have increased the degree of the modulated serie
    msa.working_degree[] = z.working_degree[]+1
    return nothing
end


# evaluation with horner method applied to the polynomial and to the Fourier serie
function (ms::TaylorFourierSeries{Float64,Float64})(t::Number)  #y(t)--> y(t,omega*t) 
    M = ms.M
    TwoMinv=0.5/M
    wdegree = ms.working_degree[]
    omega = ms.omega  #  2*pi/P
    theta = omega*t
    number_of_vars = ms.num_vars
    x=exp(im*theta)
    ret_v = Array{Float64}(undef,number_of_vars)
    @inbounds for  elem in 1:number_of_vars
        ret_v[elem]= real(ms.coeffs[1,elem,wdegree+1])  # the last element of the polynomial has constant coefficient
    end
    @inbounds for g in wdegree:-1:1 
        for elem in 1:number_of_vars
            ret_v[elem] *= t  
            xpart= ms.coeffs[M+1,elem,g]*x # initialize with the last element (with not 2*).
            for k in M:-1:2
                xpart= (xpart + 2.0*ms.coeffs[k,elem,g])*x
            end
            ret_v[elem] += real(ms.coeffs[1,elem,g]+xpart)
        end
    end
    @inbounds for  elem in 1:number_of_vars
        ret_v[elem] *=TwoMinv
    end
    return ret_v
end

# evaluation with horner method applied to the polynomial and to the Fourier serie
function (ms::TaylorFourierSeries{ComplexF64,Float64})(t::Number)  #y(t)--> y(t,omega*t) 
    M = ms.M
    wdegree = ms.working_degree[]
    omega = ms.omega  #  2*pi/P
    theta = omega*t
    number_of_vars = ms.num_vars
    inv2M = 1.0/(2*M)
    x=exp(im*theta)
    y=1/x;  # exp(-im*theta)
    ret_v = Array{ComplexF64}(undef,number_of_vars)
    @inbounds for  elem in 1:number_of_vars
        ret_v[elem]= ms.coeffs[1,elem,wdegree+1]  # the last element of the polynomial has constant coefficient
    end
    @inbounds for g in wdegree:-1:1 
        for elem in 1:number_of_vars
            ret_v[elem] *= t  # previous value multiplied by t
            # Fourier expresion: there are 2 parts:
            #   first part: sum k=1 to M  v_k *exp(im*(k-1)*theta)
            #   second part: sum k=2 to M v_{2M-k+2}*exp(-im*(k-1)*theta)  THERE is 1 element LESS
            xpart= ms.coeffs[M+1,elem,g]*x # initialize with the extra element.
            ypart = zero(ComplexF64)  # initialize with 0
            for k in M:-1:2
                xpart= (xpart + ms.coeffs[k,elem,g])*x
                ypart = (ypart+ ms.coeffs[2*M-k+2,elem,g])*y
            end
            ret_v[elem] += ms.coeffs[1,elem,g]+xpart+ypart  
        end
    end
    @inbounds for elem in 1:number_of_vars
        ret_v[elem] *= inv2M
    end
    return ret_v
end


function ComputeDiscreteSolution(ms::TaylorFourierSeries{ueltype,Float64},n_orbits::Int64) where {ueltype <: Union{Float64,ComplexF64}}
    TwoM = 2*ms.M
    wdegree = ms.working_degree[1]
    g = wdegree+1
    omega = ms.omega  #  2*pi/P ---> P = 2*pi/omega --> t= k*P/(2M)+(j-1)*P == k*pi/(omega*M)+(j-1)*P
    number_of_vars = ms.num_vars
    period = 2*π/omega
    dtheta = period/TwoM
    n_outputs = TwoM*n_orbits+1
    tt = range(0.0,stop=period*n_orbits,length=n_outputs)
    uu = Vector{Vector{ueltype}}(undef,n_outputs)
    @inbounds for i in 1:n_outputs
        uu[i] = Vector{ueltype}(undef,number_of_vars)
    end
    @inbounds for k in 0:TwoM-1
         k1 = k+1
         for j in 0:n_orbits-1
             i = j*TwoM + k1
             t= k*dtheta+j*period
             for elem in 1:number_of_vars
                 ui = ms.tcoeffs[k1,elem,g]  # the last coefficient of the polynomial 
                 for l in wdegree:-1:1
#                     ui = ms.tcoeffs[k1,elem,l] + t * ui
                    ui = muladd(ui, t, ms.tcoeffs[k1,elem,l])
                 end
                 uu[i][elem] = ui
             end
        end
    end
    t = period*n_orbits
    for elem in 1:number_of_vars
        ui = ms.tcoeffs[1,elem,g]  # the last coefficient of the polynomial 
        for l in wdegree:-1:1
            ui = muladd(ui, t, ms.tcoeffs[1,elem,l])
#            ui = ms.tcoeffs[1,elem,l] + t * ui
        end
        uu[n_outputs][elem] = ui
    end
    return collect(tt),uu
end




    

function add!(w,iw,u,iu,v,iv) 
    #w[iw] = u[iu] + v[iv]
    TwoM = 2*u.M
    g = u.working_degree[]+1
    @inbounds for j in 1:g
        for i in 1:TwoM
            w.tcoeffs[i,iw,j]= u.tcoeffs[i,iu,j]+ v.tcoeffs[i,iv,j]
        end
    end
    w.working_degree[]=u.working_degree[]
    return nothing
end



function add!(w,iw,u,iu) 
    #w[iw] = w[iu] + u[iv]
    TwoM = 2*u.M
    g = u.working_degree[]+1
    @inbounds for j in 1:g
        for i in 1:TwoM
            w.tcoeffs[i,iw,j] += u.tcoeffs[i,iu,j]
        end
    end
    w.working_degree[]=u.working_degree[]
    return nothing
end


function mult!(w,iw,u,iu,eskala)
    #w[iw] = eskala * u[iu]
    TwoM = 2*u.M
    g = u.working_degree[]+1
    @inbounds for j in 1:g
        for i in 1:TwoM
            w.tcoeffs[i,iw,j]= eskala *u.tcoeffs[i,iu,j]
        end
    end
    w.working_degree[]=u.working_degree[]
    return nothing
end

function mult!(w,iw,u,iu,v,iv)
    #w[iw] = u[iu]*v[iv]
    #   wr+ wim *i = (ur + uim*i)*(vr+vim*i) = ur*vr - uim*vim + (ur*vim + uim*vr)*i
    twoM = 2*u.M
    g = u.working_degree[]+1
    #= bi serieren arteko biderketa
           for k in 1:gmax
                aux[k,4]=0
                for m in 1:k
                    aux[k,4] +=  aux[m,1]*aux[k-m+1,2] # aux[k,4] = aux1*aux2
                end
           end
    =#
    
    @inbounds for j in 1:g
        for i in 1:twoM
            w.tcoeffs[i,iw,j] =0.0  
        end
        for m in 1:j
                # wr=ur*vr  ...
                for i in 1:twoM
                    w.tcoeffs[i,iw,j] += u.tcoeffs[i,iu,m]*v.tcoeffs[i,iv,j-m+1] 
                end
        end 
    end
    w.working_degree[]=u.working_degree[]
    return nothing
end


function mult!(u,iu::Int64,eskala)
    #w[iw] = eskala * u[iu]
    TwoM = 2*u.M
    g = u.working_degree[]+1
    @inbounds for j in 1:g
        for i in 1:TwoM
            u.tcoeffs[i,iu,j]= eskala *u.tcoeffs[i,iu,j]
        end
    end
    return nothing
end


function mult!(u,iu::Int64,eskalav::Vector{Float64})
    #w[iw] = eskala * u[iu]
    TwoM = 2*u.M
    g = u.working_degree[]+1
    @inbounds for j in 1:g
        for i in 1:TwoM
            u.tcoeffs[i,iu,j]= eskalav[i] *u.tcoeffs[i,iu,j]
        end
    end
    return nothing
end


function mult!(w, iw::Int64, u, iu::Int64, eskalav::Vector{Float64})
    #w[iw] = eskalav * u[iu]
    TwoM = 2*u.M
    g = u.working_degree[]+1
    @inbounds for j in 1:g
        for i in 1:TwoM
            w.tcoeffs[i,iw,j]= eskalav[i] *u.tcoeffs[i,iu,j]
        end
    end
    w.working_degree[]=u.working_degree[]
    return nothing
end


function mult!(u,eskala)
    twoM = 2*u.M
    g = u.working_degree[]+1
    @inbounds for j in 1:g
        for elem in 1:u.num_vars
            for i in 1:twoM
                u.tcoeffs[i,elem,j]= eskala *u.tcoeffs[i,elem,j]
            end
        end
    end
    return nothing
end


function mult!(u,eskalav::Array{Float64})
    twoM = 2*u.M
    g = u.working_degree[]+1
    @inbounds for j in 1:g
        for elem in 1:u.num_vars
            for i in 1:twoM
                u.tcoeffs[i,elem,j]= eskalav[i] *u.tcoeffs[i,elem,j]
            end
        end
    end
    return nothing
end


function mult!(w,u,v)    
    twoM = 2*u.M
    g = u.working_degree[]+1
    
    @inbounds for j in 1:g
        for elem in 1:u.num_vars
            for i in 1:twoM
                w.tcoeffs[i,elem,j] =0.0   
                for m in 1:j
                    # wr=ur*vr  ...
                    w.tcoeffs[i,elem,j] += u.tcoeffs[i,elem,m]*v.tcoeffs[i,elem,j-m+1] 
                end
            end
        end 
    end
    w.working_degree[]=u.working_degree[]
    return nothing
end

function conj!(w,iw,u,iu)
    twoM = 2*u.M
    g = u.working_degree[]+1
    @inbounds for j in 1:g
        for i in 1:twoM
            w.tcoeffs[i,iw,j] = conj(u.tcoeffs[i,iu,j])
        end 
    end
    w.working_degree[]=u.working_degree[]
    return nothing
end

function conj!(w,u)
    twoM = 2*u.M
    g = u.working_degree[]+1
    @inbounds for j in 1:g
        for elem in 1:u.num_vars
            for i in 1:twoM          
                w.tcoeffs[i,elem,j] = conj(u.tcoeffs[i,elem,j])
            end
        end
    end
    return nothing
end    


function power!(w::TaylorFourierSeries{uelType,tType},iw,u::TaylorFourierSeries{uelType,tType},iu,exponent)  where {uelType,tType}
    # w_{iw} = (u_{iu})^exponent
    TwoM = 2*u.M
    zerouel = zero(uelType)
    g = u.working_degree[]+1
    @inbounds for i in 1:TwoM
        w.tcoeffs[i,iw,1] = u.tcoeffs[i,iu,1]^exponent
        for j in 1:g-1
            aux = zerouel
            for l in 0:(j-1)
                aux = aux + (exponent*(j-l)-l)*u.tcoeffs[i,iu,j-l+1]*w.tcoeffs[i,iw,l+1]
            end
            w.tcoeffs[i,iw,j+1] = aux/(j*u.tcoeffs[i,iu,1])
        end 
    end
    w.working_degree[]=u.working_degree[]
    return nothing
end

