# Taylor-Fourier
An algorithm which provides approximate solutions to semi-linear ordinary differential equations with highly oscillatory solutions that after an appropriate change of variables can be written as a non-autonomous system with (2π/ω)-periodic dependence on t. 

The proposed approximate solutions are written in closed form as functions X(t, ω t) where X(t, θ) is:

        (i) a truncated Fourier series in θ for fixed t
        (ii) a truncated Taylor series in t for fixed θ.

The implemented method computes the approximation by combining power series arithmetic and the FFT algorithm.

Content:

  Code:

        TaylorFourier.jl (Julia code implementation)

  Examples (Jupyter notebooks):

        J2_VOP: Perturbed Kepler (satellite trajectory perturbed by J2 term or the oblateness of the earth).
        NLS : Cubic nonlinear Schrödinger equation.

