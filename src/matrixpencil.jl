# =====================================================================
# Matrix Pencil method (Hua-Sarkar) for sum-of-exponentials fitting
# =====================================================================

"""
    MatrixPencilExpansion(n=10, stepsize=1, tol=1e-8, verbosity=1)
    MatrixPencilExpansion(; n=10, stepsize=1, tol=1e-8, verbosity=1)

Parameters for the Matrix Pencil expansion algorithm: fits a data sequence to a sum of
exponentials using the matrix pencil method (Hua & Sarkar). Unlike the polynomial-type
(Prony) methods, it extracts the bases directly from the generalized eigenvalue problem
of the (truncated) Hankel matrix, which makes it considerably more robust to additive noise.
`n` is the maximum number of terms, `stepsize` the sampling step, `tol` the convergence
error and `verbosity` controls the output verbosity.
"""
struct MatrixPencilExpansion <: AbstractPronyExpansion
    n::Int
    stepsize::Int
    tol::Float64
    verbosity::Int
end
"""
    MatrixPencilExpansion(; n=10, stepsize=1, tol=1e-8, verbosity=1)

Keyword constructor for `MatrixPencilExpansion`.
"""
MatrixPencilExpansion(; n::Int=10, stepsize::Int=1, tol::Real=1.0e-8, verbosity::Int=1) =
    MatrixPencilExpansion(n, stepsize, convert(Float64, tol), verbosity)

"""
    matrix_pencil(s::Vector{<:Number}, p::Int)

Fit the data sequence `s` to a sum of `p` exponentials using the matrix pencil method.
Returns `(xs, lambdas)` such that `s(k) ≈ Σᵢ xs[i] * lambdas[i]^k` for `k = 1,2,...`.

The poles `lambdas` are obtained as the eigenvalues of `U1 \\ U2`, where `U1`/`U2` are the
truncated left singular vectors of an `R × L` Hankel matrix of `s` with its first/last row
removed, and `xs` follows from a least-squares solve with the recovered poles.
"""
function matrix_pencil(s::Vector{<:Number}, p::Int)
    N = length(s)
    T = promote_type(eltype(s), Float64)
    @assert p >= 1
    @assert p < N "p must be smaller than length(s)"

    L = clamp(div(N, 2), 1, N - 1)   # pencil parameter (number of columns)
    R = N - L + 1                    # number of rows (start positions)

    # R × L Hankel matrix
    Y = zeros(T, R, L)
    for i in 1:R, j in 1:L
        Y[i, j] = s[i + j - 1]
    end

    U = svd(Y).U                     # R × L left singular vectors
    pc = min(p, L, R - 1)
    U1 = U[1:R-1, 1:pc]              # first  R-1 rows
    U2 = U[2:R,   1:pc]              # last   R-1 rows

    z = eigvals(U1 \ U2)             # pc poles (typically real for real data)

    # least squares for the residues Rₖ with s(k-1) = Σₖ Rₖ zₖ^(k-1)
    Zpow = ones(ComplexF64, N, pc)
    zc = Vector{ComplexF64}(z)
    for k in 1:pc
        for i in 2:N
            Zpow[i, k] = zc[k]^(i - 1)
        end
    end
    Rres = Zpow \ s

    # convert to the package convention s(k) = Σᵢ xs[i] * lambdas[i]^k  (k starting at 1)
    xs = Rres ./ zc
    lambdas = zc
    return xs, lambdas
end

exponential_expansion_n(f::Vector, p::Int, alg::MatrixPencilExpansion) = matrix_pencil(f, p)