# =====================================================================
# Common interface for exponential expansion algorithms
# =====================================================================

abstract type ExponentialExpansionAlgorithm end
abstract type AbstractPronyExpansion <: ExponentialExpansionAlgorithm end

# fallback: each concrete algorithm provides its own n-term fit method
exponential_expansion_n(f::Vector, p::Int, alg::ExponentialExpansionAlgorithm) =
    throw(ArgumentError("exponential expansion not implemented for $(typeof(alg))"))

"""
    exponential_expansion(f::Vector{<:Number}, alg::ExponentialExpansionAlgorithm; stepsize=1)

Fit the data sequence `f` to a sum of exponentials, returning `(xs, lambdas)`
such that `f(k) ≈ Σᵢ xs[i] * lambdas[i]^k`.

`stepsize` (default 1) is the uniform sampling step used for the fit; the returned
coefficients/bases are rescaled back to the original sampling of `f`.
"""
function exponential_expansion(f::Vector{<:Number}, alg::ExponentialExpansionAlgorithm;
                               stepsize::Int=1)
    (length(f) > 1) || throw(ArgumentError("length of data should be larger than 1"))
    if stepsize == 1
        return _exponential_expansion_impl(f, alg)
    end
    xs, lambdas = _exponential_expansion_impl(f[1:stepsize:end], alg)
    expansion_changestepsize!(xs, lambdas, stepsize)
    (alg.verbosity > 2) && println("Exponential coefs: ", xs) &&
        println("Exponential roots: ", lambdas)
    return xs, lambdas
end

"""
    exponential_expansion_opt(f::Vector{<:Number}, alg::ExponentialExpansionAlgorithm)

Like [`exponential_expansion`](@ref), but the uniform sampling step is chosen
automatically: a cascade of candidate steps derived from the first period of `f` is
tried and the fit with the smallest reconstruction error is kept, then pruned by
[`cut`](@ref).
"""
function exponential_expansion_opt(f::Vector{<:Number}, alg::ExponentialExpansionAlgorithm)
    (length(f) > 1) || throw(ArgumentError("length of data should be larger than 1"))
    best = (Inf, nothing, nothing)          # (err, xs, lambdas)
    for step in _candidate_steps(f)
        xs, lambdas = exponential_expansion(f, alg; stepsize=step)
        err = expansion_error(f, xs, lambdas)
        if err < best[1]
            best = (err, xs, lambdas)
        end
    end
    _, xs, lambdas = best
    return cut(f, xs, lambdas, alg.tol * norm(f))
end

function expansion_changestepsize!(xs::Vector, lambdas::Vector, stepsize::Int)
    α = 1.0/stepsize
    for i in 1:length(lambdas)
        xs[i] *= (lambdas[i])^(1-α)
        lambdas[i] = (lambdas[i])^(α)
    end
    return xs, lambdas
end

function _exponential_expansion_impl(f::Vector{<:Number}, alg::ExponentialExpansionAlgorithm)
    L = length(f)
    tol = alg.tol * norm(f)
    verbosity = alg.verbosity
    maxiter = alg.n
    nitr = min(maxiter, div(L, 2))
    for n in 1:nitr
        xs, lambdas = exponential_expansion_n(f, n, alg)
        err = expansion_error(f, xs, lambdas)
        if err <= tol
            (verbosity > 1) && println("$(typeof(alg)) converged in $n iterations, error is $err")
            return xs, lambdas
        end
        if n >= min(L-n, nitr)
            (verbosity > 0) && @warn "can not find a good approximation with L=$(L), n=$(alg.n), atol=$(alg.tol), rtol=$(tol), return with error $err"
            return xs, lambdas
        end
    end
    error("can not be here")
end

function _predict(x, p)
    @assert length(p) % 2 == 0
    n = div(length(p), 2)
    L = length(x)
    T = eltype(p)
    r = zeros(T, L)
    for i in 1:L
        xi = x[i]
        @assert xi == i
        tmp = zero(T)
        for j in 1:n
            tmp += p[j] * p[n+j]^xi
        end
        r[i] = tmp
    end
    return r
end

"""
    expansion_error(f, p)

Compute the 2-norm error between the sequence predicted by the parameters `p` (coefficients and bases concatenated alternately) and the true sequence `f`.
"""
function expansion_error(f::Vector{<:Number}, p::Vector{<:Number})
    T = eltype(f)
    xdata = [convert(T, i) for i in 1:length(f)]
    f_pred = _predict(xdata, p)
    return norm(f_pred - f)
end
"""
    expansion_error(f, coeffs, alphas)

Version in which the coefficients and bases are passed separately, equivalent to `expansion_error(f, vcat(coeffs, alphas))`.
"""
expansion_error(f::Vector{<:Number}, coeffs::Vector{<:Number}, alphas::Vector{<:Number}) = expansion_error(f, vcat(coeffs, alphas))

# ----------------------------------------------------------------------
# helpers for automatic step selection ([`exponential_expansion_opt`](@ref))
# ----------------------------------------------------------------------
function first_period(x::Vector{<:Real})
    idx = findfirst(i -> !((x[i] > x[i-1]) ⊻ (x[i] > x[i+1])), 2:(length(x) - 1))
    return isnothing(idx) ? length(x) : idx + 1
end

function _candidate_steps(f::Vector{<:Number})
    idx = first_period(real(f))
    steps = unique(round.(Int, idx .* (0.2, 0.3, 0.35, 0.4, 0.45)))
    push!(steps, 1)
    filter!(s -> s > 0, steps)                        # must be a valid step size
    filter!(s -> length(f[1:s:end]) >= 2, steps)      # downsampled sequence needs ≥ 2 points
    return sort!(steps)
end

"""
    cut(f, as, bs, atol)

Keep only the largest-magnitude coefficients of the fit `(as, bs)` while the reconstruction
error on `f` stays below the absolute tolerance `atol`. Coeffs are returned sorted by
descending magnitude.
"""
function cut(f::Vector, as::Vector, bs::Vector, atol::Real)
    p = sortperm(abs.(as), rev=true)
    as, bs = as[p], bs[p]
    N = length(as)

    (expansion_error(f, as, bs) >= atol) && return as, bs
    while (expansion_error(f, as[1:N], bs[1:N]) < atol) && (N > 1)
        N -= 1
    end
    return as[1:N+1], bs[1:N+1]
end

"""
    exponential_expansion(f::Vector{<:Number}; alg=OverDeterminedProny(), stepsize=1)
    exponential_expansion(f, L::Int; alg=OverDeterminedProny(), stepsize=1)

Return coefficients `αᵢ` and bases `βᵢ` such that
f(x) = ∑ᵢ αᵢ × (βᵢ)ˣ, for 1 ≤ x ≤ N.
"""
exponential_expansion(f::Vector{<:Number}; alg::ExponentialExpansionAlgorithm=OverDeterminedProny(),
                      stepsize::Int=1) = exponential_expansion(f, alg; stepsize=stepsize)
"""
    exponential_expansion(f, L::Int, alg; stepsize=1)

Sample the function `f` on `1:L` first, then perform the exponential expansion.
"""
exponential_expansion(f, L::Int, alg::ExponentialExpansionAlgorithm; stepsize::Int=1) =
    exponential_expansion([f(k) for k in 1:L], alg; stepsize=stepsize)
exponential_expansion(f, L::Int; alg::ExponentialExpansionAlgorithm=OverDeterminedProny(),
                      stepsize::Int=1) = exponential_expansion(f, L, alg; stepsize=stepsize)

"""
    exponential_expansion_opt(f, L::Int, alg)

Sample the function `f` on `1:L` first, then perform [`exponential_expansion_opt`](@ref).
"""
exponential_expansion_opt(f, L::Int, alg::ExponentialExpansionAlgorithm) =
    exponential_expansion_opt([f(k) for k in 1:L], alg)