# =====================================================================
# LeastSquareProny: overdetermined_prony initial guess + gradient-descent refinement
# =====================================================================
#
# Fits a (real or complex) sequence to a sum of exponentials by
#   1. using overdetermined_prony to obtain an initial guess;
#   2. re-parameterizing each complex amplitude/base by its (norm, phase)
#      so the unknowns are real, then refining them with a damped
#      Gauss-Newton (Levenberg-Marquardt) solve using the analytic Jacobian.
#
# This algorithm sits on the same footing as `OverDeterminedProny` / `MatrixPencil`;
# the sampling step is carried by the `stepsize` field (default 1, or `nothing` for
# automatic selection inside [`exponential_expansion`](@ref)).

"""
    LeastSquareProny(n=10, tol=1e-8, verbosity=1, stepsize=1)
    LeastSquareProny(; n=10, tol=1e-8, verbosity=1, stepsize=1)

Parameters for the least-squares Prony + gradient-descent expansion algorithm, defined
for both real- and complex-valued sequences (real data is upcast internally). It first
obtains a least-squares Prony guess and then refines it by damped Gauss-Newton over the
real `(norm, phase)` parameterization.
`n` is the maximum number of terms, `tol` the convergence error and `verbosity` controls
the output verbosity. `stepsize` (default 1) is the uniform sampling step used by
[`exponential_expansion`](@ref); pass `stepsize=nothing` to let the algorithm select the
step automatically.
"""
struct LeastSquareProny <: ExponentialExpansionAlgorithm
    n::Int
    tol::Float64
    verbosity::Int
    stepsize::Union{Int, Nothing}
end
"""
    LeastSquareProny(; n=10, tol=1e-8, verbosity=1, stepsize=1)

Keyword constructor for `LeastSquareProny`.
"""
LeastSquareProny(; n::Int=10, tol::Real=1.0e-8, verbosity::Int=1, stepsize::Union{Int,Nothing}=1) =
    LeastSquareProny(n, convert(Float64, tol), verbosity, stepsize)

"""
    leastsquare_prony(x::Vector{<:Number}, p::Int)

Least-squares Prony fit of `x` to `p` exponentials, refined by damped
Gauss-Newton (Levenberg-Marquardt) over the `(norm, phase)` parameterization.
Real data is upcast to `ComplexF64` internally. Returns `(α, z)` with
`x(k) ≈ Σᵢ αᵢ zᵢ^k`.
"""
function leastsquare_prony(x::Vector{<:Number}, p::Int)
    f = ComplexF64.(x)
    alps, lams = overdetermined_prony(f, p)
    alps, lams, _ = lsq_expansion_n(f, alps, lams)
    return alps, lams
end

# ----------------------------------------------------------------------
# fixed-stepsize n-term fit (called by the generic iterative expansion loop)
# ----------------------------------------------------------------------
function exponential_expansion_n(f::Vector{<:Number}, p::Int, alg::LeastSquareProny)
    return leastsquare_prony(f, p)
end

# ----------------------------------------------------------------------
# re-parameterization on (norm, phase) and analytic Jacobian
#
# The complex model  f(x) = Σⱼ αⱼ λⱼˣ  is written in terms of real unknowns
#   p = [ |α|; ∠α; |λ|; ∠λ ]  (n each),
# so that  f(x) = Σⱼ |αⱼ||λⱼ|ˣ cos(∠αⱼ + x∠λⱼ) + i Σⱼ |αⱼ||λⱼ|ˣ sin(∠αⱼ + x∠λⱼ).
# ----------------------------------------------------------------------
function _lsq_predict_one(x::Integer, p::Vector{<:Real})
    n = div(length(p), 4)
    na = p[1:n]          # |α|
    pa = p[n+1:2n]        # ∠α
    nb = p[2n+1:3n]       # |λ|
    pb = p[3n+1:end]      # ∠λ

    T = eltype(p)
    rel = zero(T)
    img = zero(T)
    for j in 1:n
        tmp = na[j] * nb[j]^x
        phase = pa[j] + pb[j] * x
        rel += tmp * cos(phase)
        img += tmp * sin(phase)
    end
    return rel, img
end
function _lsq_predict(xmax::Integer, p::Vector{<:Real})
    @assert length(p) % 4 == 0
    res = [_lsq_predict_one(i, p) for i in 1:xmax]
    r = [[d[1] for d in res]; [d[2] for d in res]]
    return r
end
function _lsq_jacobian_one(x::Integer, p::Vector{<:Real})
    n = div(length(p), 4)
    na = p[1:n]          # |α|
    pa = p[n+1:2n]        # ∠α
    nb = p[2n+1:3n]       # |λ|
    pb = p[3n+1:end]      # ∠λ

    nab = @. na * nb^x
    pab = @. pa + pb * x
    nab_cos = @. nab * cos(pab)
    nab_sin = @. nab * sin(pab)

    r_na = @. nab_cos / na
    r_nb = @. nab_cos / nb * x
    r_pa = -nab_sin
    r_pb = -nab_sin * x
    i_na = @. nab_sin / na
    i_nb = @. nab_sin / nb * x
    i_pa = nab_cos
    i_pb = nab_cos * x
    return [r_na; r_pa; r_nb; r_pb], [i_na; i_pa; i_nb; i_pb]
end
function _lsq_jacobian(xmax::Integer, p::Vector{<:Real})
    @assert length(p) % 4 == 0
    res = [_lsq_jacobian_one(i, p) for i in 1:xmax]
    r = [[d[1] for d in res]; [d[2] for d in res]]
    return permutedims(hcat(r...))
end

"""
    lsq_expansion_n(f, alphas, lambdas)

Refine an initial sum-of-exponentials guess `(alphas, lambdas)` for the complex
sequence `f` by damped Gauss-Newton (Levenberg-Marquardt) minimization of the
real residual `[Re f; Im f]` over the real parameterization `[|α|; ∠α; |λ|; ∠λ]`.
Returns the refined `(alphas, lambdas, rmse)`.
"""
function lsq_expansion_n(f::Vector{<:Complex}, alphas::Vector{<:Complex}, lambdas::Vector{<:Complex})
    n = length(alphas)
    @assert n == length(lambdas)

    ydata = [real(f); imag(f)]
    xmax = length(f)
    p0 = [abs.(alphas); angle.(alphas); abs.(lambdas); angle.(lambdas)]
    predict(p) = _lsq_predict(xmax, p)
    jac(p) = _lsq_jacobian(xmax, p)
    p = _levenberg_marquardt(predict, jac, p0, ydata)

    alp = @. p[1:n] * exp(im * p[n+1:2n])
    lam = @. p[2n+1:3n] * exp(im * p[3n+1:4n])

    err = expansion_error(f, [alp; lam])
    return alp, lam, err
end

# ----------------------------------------------------------------------
# self-contained damped Gauss-Newton (Levenberg-Marquardt) least squares
# ----------------------------------------------------------------------
function _levenberg_marquardt(predict::Function, jac::Function, p0::Vector{<:Real},
                              ydata::Vector{<:Real};
                              maxiter::Int=200, λ0::Float64=1.0e-3, tol::Float64=1.0e-12)
    p = copy(p0)
    r = predict(p) .- ydata
    λ = λ0
    rscale = max(norm(ydata), one(eltype(ydata)))
    for _ in 1:maxiter
        J = jac(p)
        A = J' * J + λ * I         # J'J is symmetric positive (semi)definite
        δ = A \ (-J' * r)
        p_new = p + δ
        r_new = predict(p_new) .- ydata

        if sum(abs2, r_new) < sum(abs2, r)   # accept step, reduce damping
            p, r = p_new, r_new
            λ *= 0.5
        else                                   # reject step, increase damping
            λ *= 5.0
        end
        (norm(r) < tol * rscale) && break
    end
    return p
end