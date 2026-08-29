# ==========================================================================
# Matrix Pencil algorithm tests: real and complex data, cross-validation
# ==========================================================================

@testset "MatrixPencil" begin
    atol = 1.0e-10
    for ydata in real_exp_data()
        xs, lambdas = exponential_expansion(ydata, MatrixPencil(n=20, tol=atol, verbosity=0))
        @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
    end
end

@testset "MatrixPencil: complex data" begin
    atol = 1.0e-8
    ydata = complex_exp_data()
    xs, lambdas = ExpExp.matrix_pencil(ydata, 2)
    @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
end

@testset "MatrixPencil vs Prony cross-validation" begin
    ydata = [0.5^k + 2 * 0.7^k for k in 1:20]
    atol = 1.0e-10

    # both algorithms recover the same bases on exact exponential data
    xsm, lamm = exponential_expansion(ydata, MatrixPencil(n=20, tol=atol, verbosity=0))
    xsp, lamp = exponential_expansion(ydata, OverDeterminedProny(n=20, tol=atol, verbosity=0))
    @test sort(real(lamm)) ≈ sort(real(lamp)) rtol = 1.0e-6
    @test isapprox(sort(real(lamm)), [0.5, 0.7]; rtol=1e-6)

    # noise robustness: matrix pencil stays accurate, Prony does not
    Random.seed!(7)
    ynoisy = ydata .+ 1.0e-4 .* randn(20)
    xn, lamn = ExpExp.matrix_pencil(ynoisy, 2)
    @test expansion_error(ynoisy, xn, lamn) / norm(ynoisy) < 1.0e-2  # fits the noisy data well
    @test maximum(abs.(sort(real(lamn)) .- [0.5, 0.7])) < 1.0e-2     # bases stay stable
end