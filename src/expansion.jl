# =====================================================================
# Common interface for exponential expansion algorithms
# =====================================================================

abstract type ExponentialExpansionAlgorithm end
abstract type AbstractPronyExpansion <: ExponentialExpansionAlgorithm end

# fallback: each concrete algorithm provides its own n-term fit method
exponential_expansion_n(f::Vector, p::Int, alg::ExponentialExpansionAlgorithm) =
    throw(ArgumentError("exponential expansion not implemented for $(typeof(alg))"))

"""
    exponential_expansion(f::Vector{<:Number}, alg::AbstractPronyExpansion)

Fit the data sequence `f` to a sum of exponentials, returning `(xs, lambdas)`
such that `f(k) ≈ Σᵢ xs[i] * lambdas[i]^k`.
"""
function exponential_expansion(f::Vector{<:Number}, alg::AbstractPronyExpansion)
    (length(f) > 1) || throw(ArgumentError("length of data should be larger than 1"))
    xs, lambdas = _exponential_expansion_impl(f, alg)
    if alg.stepsize != 1
        expansion_changestepsize!(xs, lambdas, alg.stepsize)
    end
    if alg.verbosity > 2
        println("Exponential coefs: ", xs)
        println("Exponential roots: ", lambdas)
    end
    return xs, lambdas
end

function expansion_changestepsize!(xs::Vector, lambdas::Vector, stepsize::Int)
    α = 1.0/stepsize
    for i in 1:length(lambdas)
        xs[i] *= (lambdas[i])^(1-α)
        lambdas[i] = (lambdas[i])^(α)
    end
    return xs, lambdas
end

function _exponential_expansion_impl(f::Vector{<:Number}, alg::AbstractPronyExpansion)
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

"""
    exponential_expansion(f::Vector{<:Number}; alg=PronyExpansion())
    exponential_expansion(f, L::Int; alg=PronyExpansion())

Return coefficients `αᵢ` and bases `βᵢ` such that
f(x) = ∑ᵢ αᵢ × (βᵢ)ˣ, for 1 ≤ x ≤ N.
"""
exponential_expansion(f::Vector{<:Number}; alg::ExponentialExpansionAlgorithm=PronyExpansion()) = exponential_expansion(f, alg)
"""
    exponential_expansion(f, L::Int, alg)

Sample the function `f` on `1:L` first, then perform the exponential expansion.
"""
exponential_expansion(f, L::Int, alg::ExponentialExpansionAlgorithm) = exponential_expansion([f(k) for k in 1:L], alg)
exponential_expansion(f, L::Int; alg::ExponentialExpansionAlgorithm=PronyExpansion()) = exponential_expansion(f, L, alg)