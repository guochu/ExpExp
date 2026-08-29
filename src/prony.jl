# =====================================================================
# Prony expansion algorithms (least-squares and deterministic Hankel)
# =====================================================================

"""
    PronyExpansion(n=10, stepsize=1, tol=1e-8, verbosity=1)
    PronyExpansion(; n=10, stepsize=1, tol=1e-8, verbosity=1)

Parameters for the Prony expansion algorithm: fits a data sequence to a sum of exponentials using the least-squares Prony method.
`n` is the maximum number of terms, `stepsize` the sampling step, `tol` the convergence error, and `verbosity` controls the output verbosity.
"""
struct PronyExpansion <: AbstractPronyExpansion
    n::Int
    stepsize::Int
    tol::Float64
    verbosity::Int
end
"""
    PronyExpansion(; n=10, stepsize=1, tol=1e-8, verbosity=1)

Keyword constructor for `PronyExpansion`.
"""
PronyExpansion(; n::Int=10, stepsize::Int=1, tol::Real = 1.0e-8, verbosity::Int=1) = PronyExpansion(n, stepsize, convert(Float64, tol), verbosity)

"""
    DeterminedPronyExpansion(n=10, stepsize=1, tol=1e-8, verbosity=1)
    DeterminedPronyExpansion(; n=10, stepsize=1, tol=1e-8, verbosity=1)

Parameters for the deterministic Prony expansion algorithm: fits a data sequence to a sum of exponentials using the exact Hankel method (rather than least squares).
The parameters have the same meaning as in `PronyExpansion`.
"""
struct DeterminedPronyExpansion <: AbstractPronyExpansion
    n::Int
    stepsize::Int
    tol::Float64
    verbosity::Int
end
"""
    DeterminedPronyExpansion(; n=10, stepsize=1, tol=1e-8, verbosity=1)

Keyword constructor for `DeterminedPronyExpansion`.
"""
DeterminedPronyExpansion(; n::Int=10, stepsize::Int=1, tol::Real = 1.0e-8, verbosity::Int=1) = DeterminedPronyExpansion(n, stepsize, convert(Float64, tol), verbosity)


"""
    prony(x::Vector, p::Int)

Exact Prony fit of `x` to `p` exponentials using the (deterministic) Hankel method. Returns `(α, z)` with `x(k) ≈ Σᵢ αᵢ zᵢ^k`.
"""
function prony(x::Vector, p::Int)
    n = length(x)
    @assert p <= n÷2 "p can not exceed length(x)/2"

    # find the roots of characteristic polynomial
    A = zeros(typeof(x[1]), p, p)
    for i = 1:p, j = 1:p
        A[i,j] = x[p+i-j]
    end
    a = -A\x[p+1:2p]
    pushfirst!(a,1)
    z = roots(Polynomial(reverse(a)))

    # find the coefficient
    A = zeros(typeof(z[1]), p, p)
    for i = 1:p, j = 1:p
        A[i,j] = z[j]^i
    end
    α = A\x[1:p]

    α, z
end

"""
    lsq_prony(x::Vector, p::Int)

Least-squares Prony fit of `x` to `p` exponentials. Returns `(α, z)` with `x(k) ≈ Σᵢ αᵢ zᵢ^k`.
More robust than `prony` when the data is noisy or the number of terms is not exactly `p`.
"""
function lsq_prony(x::Vector, p::Int)
    n = length(x)
    @assert p <= n÷2 "p can not exceed length(x)/2"

    # find the roots of characteristic polynomial via least square method
    A = zeros(typeof(x[1]), n-p , p)
    for i = 1:n-p, j = 1:p
        A[i,j] = x[p+i-j]
    end
    a = -A\x[p+1:n]
    pushfirst!(a,1)
    z = roots(Polynomial(reverse(a)))

    # find the coefficient via least square method
    A = zeros(typeof(z[1]), n, p)
    for i = 1:n, j = 1:p
        A[i,j] = z[j]^i
    end
    α = A\x

    α, z
end

exponential_expansion_n(f::Vector, p::Int, alg::PronyExpansion) = lsq_prony(f, p)
exponential_expansion_n(f::Vector, p::Int, alg::DeterminedPronyExpansion) = prony(f, p)