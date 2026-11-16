# =====================================================================
# Anchored Matrix Pencil: row-subsampled, two-end-anchored Hua–Sarkar
# =====================================================================

# Subsampled Hankel pair of `y` (uniform row selection up to `pencil_rows` rows)
# with the compact SVD of the shifted matrix H0. Returns `(U, S, Vh, H1)` where
# `Vh` is the adjoint of the right singular vectors and `H1` is the one-step
# shifted Hankel. The column space of the pencil is the Vandermonde span of the
# poles, so the row selection does not change it.
function _anchored_pencil_init(y::Vector{ComplexF64}, pencil_rows::Int)
    L = length(y)
    h = L ÷ 2
    rows = h <= pencil_rows ? collect(0:h-1) : unique(round.(Int, range(0, h - 1; length=pencil_rows)))
    H0 = [y[i+j] for i in rows, j in 1:L-h]
    H1 = [y[i+j+1] for i in rows, j in 1:L-h]
    F = svd(H0)
    return F.U, F.S, F.Vt, H1
end

# Fit a single rank `r`: pencil eigenvalues from the rank-`r` projection, folded
# to (nodes, anchors) with all |node| ≤ 1, followed by a truncated-SVD
# least-squares solve for the weights. Returns `(nodes, anchors, w, err)`, or
# `nothing` if the pencil eigenvalues are nonfinite at this rank.
function _anchored_pencil_rank(U, S, Vh, H1, y::Vector{ComplexF64}, r::Int, rightanchors::Bool)
    L = length(y)
    A = (U[:, 1:r]' * H1 * Vh[1:r, :]') * Diagonal(1 ./ S[1:r])
    nodes = ComplexF64.(eigvals(A))
    all(isfinite, nodes) || return nothing
    # without anchoring, out-of-disc poles are projected onto the unit circle;
    # with two-end anchoring they are folded to :right anchors with node 1/q.
    nodes = [!rightanchors && abs(q) > 1 ? q / abs(q) : q for q in nodes]
    anchors = [rightanchors && abs(q) > 1 ? :right : :left for q in nodes]
    nodes = [a == :right ? inv(q) : q for (a, q) in zip(anchors, nodes)]
    B = ComplexF64[q^(a == :left ? d - 1 : L - d) for d in 1:L, (q, a) in zip(nodes, anchors)]
    F = svd(B)
    sv = F.S
    keep = sv .> maximum(sv) * 1e-14
    w = F.Vt[keep, :]' * ((F.U[:, keep]' * y) ./ sv[keep])
    err = norm(B * w - y) / norm(y)
    return nodes, anchors, w, err
end

"""
    anchored_matrix_pencil(s::Vector{<:Number}, p::Int; pencil_rows=128, rightanchors=true)
        -> (xs, lambdas, anchors)

Hua–Sarkar matrix pencil fit of the data sequence `s` to `p` exponentials with
uniform row subsampling of the Hankel matrix to `pencil_rows` rows (the pencil
column space is the Vandermonde span of the poles, so row selection does not
change it) and two-end anchoring. Returns `(xs, lambdas, anchors)` reproducing

```
s(d) ≈ Σᵢ xs[i] * lambdas[i]^(d - 1)   if anchors[i] == :left
s(d) ≈ Σᵢ xs[i] * lambdas[i]^(L - d)   if anchors[i] == :right
```

on the sampling window `d = 1,...,L = length(s)`, with all `|lambdas[i]| ≤ 1`:
poles `q` with `|q| > 1` are folded to `:right` anchors with node `1/q`, which
is the exact identity `q^(d-1) = q^(L-1) * (1/q)^(L-d)` with the factor absorbed
in `xs`. With `rightanchors=false` such poles are projected onto the unit circle
(`q/|q|`) instead and all anchors are `:left`.

Throws an `ArgumentError` if the pencil is rank deficient or produces nonfinite
eigenvalues at the requested order.
"""
function anchored_matrix_pencil(s::Vector{<:Number}, p::Int; pencil_rows::Int=128,
                                rightanchors::Bool=true)
    pencil_rows >= 1 || throw(ArgumentError("pencil_rows must be >= 1"))
    p >= 1 || throw(ArgumentError("p must be >= 1"))
    y = ComplexF64.(s)
    all(isfinite, y) || throw(ArgumentError("nonfinite fit samples"))
    length(y) >= 2 || throw(ArgumentError("need at least two fit samples"))
    U, S, Vh, H1 = _anchored_pencil_init(y, pencil_rows)
    r = min(p, length(S))
    S[r] > 0 || throw(ArgumentError("matrix pencil rank deficient at order $p"))
    res = _anchored_pencil_rank(U, S, Vh, H1, y, r, rightanchors)
    res === nothing && throw(ArgumentError("matrix pencil produced nonfinite eigenvalues at order $p"))
    nodes, anchors, w = res
    return w, nodes, anchors
end

"""
    anchored_matrix_pencil_sweep(s::Vector{<:Number}; tol=1e-10, n=40, pencil_rows=128,
                                 rightanchors=true) -> (xs, lambdas, anchors, err)

Minimal-order sweep over the anchored matrix pencil: the rank-`r` fit of
[`anchored_matrix_pencil`](@ref) is evaluated for `r = 1,...,n` reusing a single
SVD of the subsampled Hankel matrix, and the first rank whose relative
reconstruction error `err = ‖B*w - s‖ / ‖s‖` satisfies `err ≤ tol` is returned.
Throws an `ArgumentError` carrying the best residual if no rank qualifies.

This is semantically equivalent to calling `anchored_matrix_pencil(s, r)` for
`r = 1,...,n` and keeping the first fit under tolerance, but at a single pencil
SVD cost; performance-sensitive consumers should prefer this function over the
`AnchoredMatrixPencil` algorithm structure (which refits — and re-`svd`s — the
Hankel matrix at every order through the shared expansion loop).
"""
function anchored_matrix_pencil_sweep(s::Vector{<:Number}; tol::Real=1.0e-10, n::Int=40,
                                      pencil_rows::Int=128, rightanchors::Bool=true)
    0 < tol < 1 && n >= 1 && pencil_rows >= 1 || throw(ArgumentError("invalid fit settings"))
    y = ComplexF64.(s)
    all(isfinite, y) || throw(ArgumentError("nonfinite fit samples"))
    length(y) >= 2 || throw(ArgumentError("need at least two fit samples"))
    U, S, Vh, H1 = _anchored_pencil_init(y, pencil_rows)
    best = Inf
    for r in 1:min(n, length(S))
        S[r] > 0 || break
        res = _anchored_pencil_rank(U, S, Vh, H1, y, r, rightanchors)
        res === nothing && continue
        nodes, anchors, w, err = res
        best = min(best, err)
        err <= tol && return w, nodes, anchors, err
    end
    throw(ArgumentError("anchored matrix-pencil fit residual $best exceeds $tol with n=$n"))
end

"""
    AnchoredMatrixPencil(; n=40, tol=1e-10, verbosity=1, pencil_rows=128, rightanchors=true,
                         stepsize=1)

Parameters for the anchored matrix pencil expansion algorithm: the
[`MatrixPencil`](@ref) method augmented with uniform Hankel row subsampling to
`pencil_rows` rows and two-end anchoring, exposing the fit as an
[`ExponentialExpansionAlgorithm`](@ref) with the common `(xs, lambdas)` return
convention. The default parameter values match the influence-function fit of
GTEMPO (`maxterms=40`, `rtol=1e-10`, `pencil_rows=128`, `rightanchors=true`).

!!! note
    The shared expansion loop refits (and re-`svd`s) the Hankel matrix at every
    order; for the single-SVD rank sweep use
    [`anchored_matrix_pencil_sweep`](@ref) directly.
"""
struct AnchoredMatrixPencil <: ExponentialExpansionAlgorithm
    n::Int
    tol::Float64
    verbosity::Int
    pencil_rows::Int
    rightanchors::Bool
    stepsize::Union{Int, Nothing}
end
"""
    AnchoredMatrixPencil(; n=40, tol=1e-10, verbosity=1, pencil_rows=128, rightanchors=true,
                         stepsize=1)

Keyword constructor for `AnchoredMatrixPencil`.
"""
AnchoredMatrixPencil(; n::Int=40, tol::Real=1.0e-10, verbosity::Int=1, pencil_rows::Int=128,
                     rightanchors::Bool=true, stepsize::Union{Int,Nothing}=1) =
    AnchoredMatrixPencil(n, convert(Float64, tol), verbosity, pencil_rows, rightanchors, stepsize)

# Flatten the anchored triple to the package convention `f(k) = Σᵢ xs[i]*λ[i]^k`:
# a `:left` term `xs*q^(k-1)` maps to `(xs/q, q)` and a `:right` term `xs*q^(L-k)`
# maps to `(xs*q^L, 1/q)` (the flattening is value-identical on the fitted window
# `k = 1,...,L`; `|λ| > 1` flags a `:right` anchor with node `1/λ`). As for
# `MatrixPencil`, a vanishing node is not representable in the flattened
# convention (division by zero).
function _anchored_pencil_flatten(xs, lambdas, anchors, L::Int)
    xs2 = Vector{ComplexF64}(undef, length(xs))
    lambdas2 = Vector{ComplexF64}(undef, length(xs))
    for i in eachindex(xs)
        if anchors[i] === :left
            xs2[i] = xs[i] / lambdas[i]
            lambdas2[i] = lambdas[i]
        else
            xs2[i] = xs[i] * lambdas[i]^L
            lambdas2[i] = inv(lambdas[i])
        end
    end
    return xs2, lambdas2
end

function exponential_expansion_n(f::Vector, p::Int, alg::AnchoredMatrixPencil)
    L = length(f)
    xs, lambdas, anchors = anchored_matrix_pencil(f, p; pencil_rows=alg.pencil_rows,
                                                  rightanchors=alg.rightanchors)
    return _anchored_pencil_flatten(xs, lambdas, anchors, L)
end
