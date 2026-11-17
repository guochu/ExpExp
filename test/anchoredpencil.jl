# ==========================================================================
# Anchored matrix pencil tests: row subsampling, two-end anchoring, sweep
# ==========================================================================

# Evaluate an anchored triple (xs, lambdas, anchors) on its fitting window d = 1..L:
# left terms contribute xs*λ^(d-1), right terms contribute xs*λ^(L-d).
anchored_eval(xs, lambdas, anchors, L) =
    [ComplexF64(sum(xs[i] * (anchors[i] === :left ? lambdas[i]^(d - 1) : lambdas[i]^(L - d))
                        for i in eachindex(xs)))
     for d in 1:L]

# Original poles of an anchored triple: fold :right nodes back through 1/λ.
anchored_poles(lambdas, anchors) =
    [a === :right ? inv(q) : q for (q, a) in zip(lambdas, anchors)]

# clean k-term complex signal on 1:L
exp_signal(qs, coeffs, L) = ComplexF64[sum(coeffs[i] * qs[i]^k for i in eachindex(qs)) for k in 1:L]

@testset "AnchoredMatrixPencil: defaults" begin
    alg = AnchoredMatrixPencil()
    @test alg.n == 40
    @test alg.tol == 1e-10
    @test alg.verbosity == 1
    @test alg.pencil_rows == 128
    @test alg.stepsize == 1
end

@testset "anchored_matrix_pencil: exact recovery" begin
    atol = 1.0e-8
    L = 40

    # decaying poles with complex phases, all |q| < 1: left anchors only
    qs = [0.5, 0.7im, 0.9 * cis(0.4)]
    coeffs = [1.0 + 2.0im, -0.5 + 0.3im, 2.0 - 1.0im]
    y = exp_signal(qs, coeffs, L)
    xs, lambdas, anchors = anchored_matrix_pencil(y, 3)
    @test all(a === :left for a in anchors)
    @test all(abs.(lambdas) .<= 1)
    @test anchored_eval(xs, lambdas, anchors, L) ≈ y atol = atol * norm(y)
    @test sort(anchored_poles(lambdas, anchors), by=abs) ≈ sort(qs, by=abs) rtol = 1e-6 atol = 1e-8

    # two-sided window: growing pole |q| > 1 folds to a :right anchor with node 1/q
    qs = [1.2, 0.6, 0.9im]
    coeffs = [0.3 - 0.7im, 1.5 + 0.2im, -1.0im]
    y = exp_signal(qs, coeffs, L)
    xs, lambdas, anchors = anchored_matrix_pencil(y, 3)
    @test count(a === :right for a in anchors) == 1
    @test all(abs.(lambdas) .<= 1)
    @test anchored_eval(xs, lambdas, anchors, L) ≈ y atol = atol * norm(y)
    @test sort(anchored_poles(lambdas, anchors), by=abs) ≈ sort(qs, by=abs) rtol = 1e-6 atol = 1e-8
end

@testset "anchored_matrix_pencil: fold identity on minimal window" begin
    # L = 2: growing data node folds to a right anchor and reproduces the samples exactly
    y = ComplexF64[1.0, 2.0]
    xs, lambdas, anchors = anchored_matrix_pencil(y, 1)
    @test anchors == [:right]
    @test lambdas[1] ≈ 0.5 rtol = 1e-12
    @test anchored_eval(xs, lambdas, anchors, 2) ≈ y atol = 1e-12
end

@testset "anchored_matrix_pencil: row subsampling vs full rows" begin
    # L = 600 with h = 300 > pencil_rows = 128 triggers subsampling; on clean data
    # the pencil column space is unchanged, so the poles agree with the full-row fit
    L = 600
    qs = [0.8 * cis(0.3), 0.5 + 0.1im, 0.9 * cis(-1.1), 0.3, 0.6 * cis(2.0)]
    coeffs = [randn(ComplexF64) for _ in 1:5]
    y = exp_signal(qs, coeffs, L)
    xs, lambdas, anchors = anchored_matrix_pencil(y, 5; pencil_rows=128)
    xs2, lambdas2, anchors2 = anchored_matrix_pencil(y, 5; pencil_rows=300)   # h <= rows: full
    @test sort(anchored_poles(lambdas, anchors), by=abs) ≈
          sort(anchored_poles(lambdas2, anchors2), by=abs) rtol = 1e-6 atol = 1e-8
    @test anchored_eval(xs, lambdas, anchors, L) ≈ y atol = 1e-8 * norm(y)
end

@testset "anchored_matrix_pencil_sweep: minimal order and rtol contract" begin
    L = 128
    qs = [0.7 * cis(0.3), 0.5 + 0.1im, 0.9 * cis(-1.1), 0.6 * cis(2.0)]
    coeffs = [1.0 + 0.5im, -0.8 + 1.2im, 0.4 - 0.3im, 1.1im]
    y = exp_signal(qs, coeffs, L)
    tol = 1.0e-10

    xs, lambdas, anchors, err = anchored_matrix_pencil_sweep(y; tol=tol, n=20)
    @test length(lambdas) == 4          # minimal order: stops at the first rank under tol
    @test err <= tol
    @test all(abs.(lambdas) .<= 1)
    @test anchored_eval(xs, lambdas, anchors, L) ≈ y atol = 1e-10 * norm(y)

    # sweep is bitwise-equivalent to per-order calls of the single-order fit
    xs1, lambdas1, anchors1 = anchored_matrix_pencil(y, 4)
    @test all(xs .=== xs1) && all(lambdas .=== lambdas1) && anchors == anchors1

    # unreachable tolerance: ArgumentError carrying the best residual
    @test_throws ArgumentError anchored_matrix_pencil_sweep(y; tol=1.0e-300, n=40)
    try
        anchored_matrix_pencil_sweep(y; tol=1.0e-300, n=40)
    catch e
        @test occursin("residual", e.msg)
    end
end

@testset "supercritical poles are always folded, never projected" begin
    # the only handling of |q| > 1 is the exact :right fold — no silent
    # projection onto the unit circle exists in this algorithm
    L = 40
    qs = [1.2, 0.6]
    coeffs = [0.3 - 0.7im, 1.5 + 0.2im]
    y = exp_signal(qs, coeffs, L)
    xs, lambdas, anchors = anchored_matrix_pencil(y, 2)
    @test count(a === :right for a in anchors) == 1   # eigvals order is not deterministic
    @test count(a === :left for a in anchors) == 1
    @test all(abs.(lambdas) .<= 1)
    @test anchored_eval(xs, lambdas, anchors, L) ≈ y atol = 1e-8 * norm(y)
    @test sort(anchored_poles(lambdas, anchors), by=abs) ≈ sort(qs, by=abs) rtol = 1e-6 atol = 1e-8
end

@testset "anchored triple <-> flattened round-trip" begin
    L = 64
    qs = [1.15 * cis(0.4), 0.7, 0.85im, 0.4 + 0.2im]
    coeffs = [randn(ComplexF64) for _ in 1:4]
    y = exp_signal(qs, coeffs, L)
    xs, lambdas, anchors = anchored_matrix_pencil(y, 4)

    # flatten (package convention) reproduces the anchored fit on the window
    fxs, flambdas = ExpExp._anchored_pencil_flatten(xs, lambdas, anchors, L)
    @test [sum(fxs[i] * flambdas[i]^k for i in eachindex(fxs)) for k in 1:L] ≈
          anchored_eval(xs, lambdas, anchors, L) atol = 1e-10
    @test expansion_error(y, fxs, flambdas) / norm(y) < 1e-10

    # bijection: |λ| > 1 flags a :right anchor with node 1/λ; restore and compare
    restored_xs = [a === :left ? x * q : x * q^L for (x, q, a) in zip(fxs, flambdas, anchors)]
    @test restored_xs ≈ xs rtol = 1e-12
    restored_lambdas = [a === :left ? q : inv(q) for (q, a) in zip(flambdas, anchors)]
    @test restored_lambdas ≈ lambdas rtol = 1e-12
    @test anchors == [abs(q) > 1 ? :right : :left for q in flambdas]
end

@testset "AnchoredMatrixPencil: shared expansion loop" begin
    atol = 1.0e-10
    # real data through the shared loop (flattened convention + iszero guard)
    ydata = [0.5^k + 2 * 0.7^k for k in 1:40]
    xs, lambdas = exponential_expansion(ydata, AnchoredMatrixPencil(n=20, tol=atol, verbosity=0))
    @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
    @test length(lambdas) == 2

    # complex data with a growing mode: flat |λ| > 1 term is value-identical on the window
    y = exp_signal([1.2, 0.6], [0.3 - 0.7im, 1.5 + 0.2im], 40)
    xs, lambdas = exponential_expansion(y, AnchoredMatrixPencil(n=10, tol=1e-9, verbosity=0))
    @test count(abs.(lambdas) .> 1) == 1
    @test expansion_error(y, xs, lambdas) / norm(y) < 1e-9

    # degenerate input: the shared-loop iszero guard returns the trivial zero term
    xs, lambdas = exponential_expansion(zeros(ComplexF64, 6), AnchoredMatrixPencil(n=3, verbosity=0))
    @test all(iszero, xs) && all(iszero, lambdas)
end

@testset "anchored pencil: argument guards" begin
    y = ComplexF64[1.0, 2.0, 3.0, 4.0]
    @test_throws ArgumentError anchored_matrix_pencil(y, 0)
    @test_throws ArgumentError anchored_matrix_pencil(y, 1; pencil_rows=0)
    @test_throws ArgumentError anchored_matrix_pencil(ComplexF64[1.0], 1)
    @test_throws ArgumentError anchored_matrix_pencil(ComplexF64[1.0, NaN], 1)
    @test_throws ArgumentError anchored_matrix_pencil_sweep(y; tol=0.0)
    @test_throws ArgumentError anchored_matrix_pencil_sweep(y; tol=1.0)
    @test_throws ArgumentError anchored_matrix_pencil_sweep(y; n=0)
    @test_throws ArgumentError anchored_matrix_pencil_sweep(y; pencil_rows=0)
    @test_throws ArgumentError anchored_matrix_pencil_sweep(ComplexF64[1.0]; tol=1e-10)
end
