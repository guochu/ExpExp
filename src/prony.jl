# =====================================================================
# Prony expansion algorithms (least-squares and deterministic Hankel)
# =====================================================================

"""
    OverDeterminedProny(n=10, tol=1e-8, verbosity=1)
    OverDeterminedProny(; n=10, tol=1e-8, verbosity=1)

Parameters for the Prony expansion algorithm: fits a data sequence to a sum of exponentials using the least-squares Prony method.
`n` is the maximum number of terms, `tol` the convergence error, and `verbosity` controls the output verbosity.
The sampling step is supplied via the `stepsize` keyword of [`exponential_expansion`](@ref) (default 1).
"""
struct OverDeterminedProny <: AbstractPronyExpansion
    n::Int
    tol::Float64
    verbosity::Int
end
"""
    OverDeterminedProny(; n=10, tol=1e-8, verbosity=1)

Keyword constructor for `OverDeterminedProny`.
"""
OverDeterminedProny(; n::Int=10, tol::Real = 1.0e-8, verbosity::Int=1) = OverDeterminedProny(n, convert(Float64, tol), verbosity)

"""
    DeterminedProny(n=10, tol=1e-8, verbosity=1)
    DeterminedProny(; n=10, tol=1e-8, verbosity=1)

Parameters for the deterministic Prony expansion algorithm: fits a data sequence to a sum of exponentials using the exact Hankel method (rather than least squares).
The parameters have the same meaning as in `OverDeterminedProny`.
"""
struct DeterminedProny <: AbstractPronyExpansion
    n::Int
    tol::Float64
    verbosity::Int
end
"""
    DeterminedProny(; n=10, tol=1e-8, verbosity=1)

Keyword constructor for `DeterminedProny`.
"""
DeterminedProny(; n::Int=10, tol::Real = 1.0e-8, verbosity::Int=1) = DeterminedProny(n, convert(Float64, tol), verbosity)


"""
    determined_prony(x::Vector, p::Int)

Exact Prony fit of `x` to `p` exponentials using the (deterministic) Hankel method. Returns `(α, z)` with `x(k) ≈ Σᵢ αᵢ zᵢ^k`.
"""
function determined_prony(x::Vector, p::Int)
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
    overdetermined_prony(x::Vector, p::Int)

Least-squares Prony fit of `x` to `p` exponentials. Returns `(α, z)` with `x(k) ≈ Σᵢ αᵢ zᵢ^k`.
More robust than `determined_prony` when the data is noisy or the number of terms is not exactly `p`.
"""
function overdetermined_prony(x::Vector, p::Int)
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

exponential_expansion_n(f::Vector, p::Int, alg::OverDeterminedProny) = overdetermined_prony(f, p)
exponential_expansion_n(f::Vector, p::Int, alg::DeterminedProny) = determined_prony(f, p)