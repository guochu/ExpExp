# ==========================================================================
# Refinement of Prony / lsq-Prony fits by gradient-based nonlinear least squares
#
# Strategy (as in GTEMPO's exponentialexpansion2.jl):
#   1. use (multi-step) prony / lsq_prony to obtain an initial guess;
#   2. re-parameterize each complex amplitude/base by its (norm, phase) so the
#      unknowns are real, then refine them with a damped Gauss-Newton
#      (Levenberg-Marquardt) solver using the analytic Jacobian.
#
# Two new algorithm types are provided:
#   * PronyExpansion2 : deterministic lsq_prony over several downsampling steps
#   * LsqExpansion2   : lsq_prony initial guess + gradient-descent refinement
#                       (complex-valued data only)
# ==========================================================================

struct PronyExpansion2 <: AbstractPronyExpansion
    n::Int
    atol::Float64
    rtol::Float64
    verbosity::Int
end
PronyExpansion2(; n::Int=10, atol::Real=1.0e-8, rtol::Real=1.0e-3, verbosity::Int=1) =
    PronyExpansion2(n, convert(Float64, atol), convert(Float64, rtol), verbosity)
Base.similar(alg::PronyExpansion2; n::Int=alg.n, atol::Real=alg.atol, rtol::Real=alg.rtol, verbosity::Int=alg.verbosity) =
    PronyExpansion2(n=n, atol=atol, rtol=rtol, verbosity=verbosity)

struct LsqExpansion2 <: ExponentialExpansionAlgorithm
    n::Int
    atol::Float64
    rtol::Float64
    verbosity::Int
end
LsqExpansion2(; n::Int=10, atol::Real=1.0e-8, rtol::Real=1.0e-3, verbosity::Int=1) =
    LsqExpansion2(n, convert(Float64, atol), convert(Float64, rtol), verbosity)
Base.similar(alg::LsqExpansion2; n::Int=alg.n, atol::Real=alg.atol, rtol::Real=alg.rtol, verbosity::Int=alg.verbosity) =
    LsqExpansion2(n=n, atol=atol, rtol=rtol, verbosity=verbosity)

const ExponentialExpansionAlgorithm2 = Union{PronyExpansion2, LsqExpansion2}

# --------------------------------------------------------------------------
# main entry: tune the tolerance, pick cascade of downsampling steps
# --------------------------------------------------------------------------
function exponential_expansion(f::Vector{<:Number}, alg::ExponentialExpansionAlgorithm2)
    r_atol = norm(f) * alg.rtol
    if r_atol < alg.atol
        (alg.verbosity >= 1) && println("Using atol of $r_atol according to rtol")
        alg = similar(alg, atol=r_atol)
    end

    idx = first_period(real(f))
    steps = unique([[round(Int, idx * i) for i in [0.2, 0.3, 0.35, 0.4, 0.45]]; 1])
    filter!(x -> (x > 0), steps)
    xs, lambdas = _exponential_expansion(f, alg, steps=steps)
    xs, lambdas = cut(f, xs, lambdas, alg)
    if alg.verbosity >= 2
        println("Prony coefs: ", xs)
        println("Prony roots: ", lambdas)
    end
    return xs, lambdas
end

# --------------------------------------------------------------------------
# PronyExpansion2: deterministic multi-step fit
# --------------------------------------------------------------------------
function _exponential_expansion(f::Vector{<:Number}, alg::PronyExpansion2; steps::Vector{Int})
    L = length(f)
    atol = alg.atol
    nitr = min(alg.n, L)
    for n in 1:nitr
        (xs, lambdas), err = exponential_expansion_n(f, n, alg, steps=steps)
        if err <= atol
            (alg.verbosity >= 1) && println("PronyExpansion2 converged in $n iterations, error is $err")
            return xs, lambdas
        end
        if n >= min(L - n + 1, nitr)
            @warn "can not find a good approximation with L=$(L), n=$(alg.n), atol=$(atol), return with error $err"
            return xs, lambdas
        end
    end
    error("can not be here")
end

function exponential_expansion_n(f::Vector, p::Int, alg::PronyExpansion2; steps::Vector{Int})
    errs = []
    coeffs = []
    for step in steps
        f2 = f[1:step:end]
        (p > length(f2) ÷ 2) && continue

        xs, lambdas = lsq_prony(f2, p)
        xs′ = @. xs * lambdas^(1 - 1 / step)
        lambdas′ = @. lambdas^(1 / step)
        err = expansion_error(f, xs′, lambdas′)

        push!(errs, err)
        push!(coeffs, (xs′, lambdas′))
    end

    _, idx = findmin(identity, errs)
    return coeffs[idx], errs[idx]
end

# --------------------------------------------------------------------------
# LsqExpansion2: lsq_prony initial guess + gradient-descent refinement
# --------------------------------------------------------------------------
_exponential_expansion(f::Vector{<:Real}, alg::LsqExpansion2; steps::Vector{Int}) =
    error("LsqExpansion2 only support Complex data currently")

function _exponential_expansion(f::Vector{<:Complex}, alg::LsqExpansion2; steps::Vector{Int})
    L = length(f)
    atol = alg.atol
    nitr = min(alg.n, L)

    alps, lams = lsq_prony(f, 1)
    err = expansion_error(f, [alps; lams])
    best_n, best_err, best_alps, best_lams = 1, err, alps, lams

    for n in 1:nitr
        (alps, lams), err = _lsq_expansion_n(f, n, steps=steps)
        if err < best_err
            best_n, best_alps, best_lams, best_err = n, alps, lams, err
        end
        if best_err <= atol
            (alg.verbosity >= 1) && println("LsqExpansion2 converged in $n iterations, error is $best_err")
            return best_alps, best_lams
        end
        if n >= min(L - n + 1, nitr)
            @warn "can not find a good approximation with L=$(L), n=$(alg.n), atol=$(atol), return with error $err"
            return best_alps, best_lams
        end
    end
    error("can not be here")
end

function _lsq_expansion_n(f::Vector{<:Complex}, p::Int; steps::Vector{Int})
    errs = []
    coeffs = []
    for step in steps
        f2 = f[1:step:end]
        (p > length(f2) ÷ 2) && continue

        xs, lambdas = lsq_prony(f2, p)
        xs′ = @. xs * lambdas^(1 - 1 / step)
        lambdas′ = @. lambdas^(1 / step)

        xs′, lambdas′, err = lsq_expansion_n(f, xs′, lambdas′)
        err = expansion_error(f, xs′, lambdas′)

        push!(errs, err)
        push!(coeffs, (xs′, lambdas′))
    end

    _, idx = findmin(identity, errs)
    return coeffs[idx], errs[idx]
end

# --------------------------------------------------------------------------
# re-parameterization on (norm, phase) and analytic Jacobian
#
# The complex model  f(x) = Σⱼ αⱼ λⱼˣ  is written in terms of real unknowns
#   p = [ |α|; ∠α; |λ|; ∠λ ]  (n each),
# so that  f(x) = Σⱼ |αⱼ||λⱼ|ˣ cos(∠αⱼ + x∠λⱼ) + i Σⱼ |αⱼ||λⱼ|ˣ sin(∠αⱼ + x∠λⱼ).
# --------------------------------------------------------------------------
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
real repSidual `[Re f; Im f]` over the log-parameterization `[|α|; ∠α; |λ|; ∠λ]`.
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

# --------------------------------------------------------------------------
# self-contained damped Gauss-Newton (Levenberg-Marquardt) least squares
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
function first_period(x::Vector{<:Real})
    idx = findfirst(i -> !((x[i] > x[i-1]) ⊻ (x[i] > x[i+1])), 2:(length(x) - 1))
    return isnothing(idx) ? length(x) : idx + 1
end

function cut(f::Vector, as::Vector, bs::Vector, alg::ExponentialExpansionAlgorithm2)
    p = sortperm(abs.(as), rev=true)
    as, bs = as[p], bs[p]
    N = length(as)

    (expansion_error(f, as, bs) >= alg.atol) && return as, bs
    while (expansion_error(f, as[1:N], bs[1:N]) < alg.atol) && (N > 1)
        N -= 1
    end
    return as[1:N+1], bs[1:N+1]
end