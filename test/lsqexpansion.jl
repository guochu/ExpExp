# ==========================================================================
# Automatic-step (exponential_expansion_opt) and LeastSquareProny tests
# ==========================================================================

@testset "exponential_expansion_opt: real data" begin
    atol = 1.0e-8
    for alg in (OverDeterminedProny(n=10, tol=atol), MatrixPencil(n=10, tol=atol))
        for ydata in real_exp_data()
            xs, lambdas = exponential_expansion_opt(ydata, alg)
            @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
        end
    end
end

@testset "exponential_expansion_opt: complex data" begin
    atol = 1.0e-8
    ydata = complex_exp_data()
    for alg in (OverDeterminedProny(n=10, tol=atol),
                MatrixPencil(n=10, tol=atol),
                LeastSquareProny(n=10, tol=atol))
        xs, lambdas = exponential_expansion_opt(ydata, alg)
        @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
    end
end

@testset "LeastSquareProny: fixed stepsize (complex)" begin
    atol = 1.0e-8
    ydata = complex_exp_data()
    xs, lambdas = exponential_expansion(ydata, LeastSquareProny(n=10, tol=atol))
    @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
end

@testset "LeastSquareProny: rejects real data" begin
    @test_throws ErrorException exponential_expansion(real_exp_data()[1], LeastSquareProny())
end

@testset "lsq_expansion_n: direct refinement" begin
    # start from an imperfect (slightly perturbed) exact guess and refine it
    ydata = complex_exp_data()
    exact = [(2.0 + 1.0im), (0.5 + 0.3im), (1.0 - 2.0im), (-0.4 + 0.6im)]
    α0 = [exact[1] * (1 + 0.02im); exact[3] * (1 - 0.02im)]
    λ0 = [exact[2] * (1 + 0.03); exact[4] * (1 - 0.03)]
    α, λ, err = ExpExp.lsq_expansion_n(ydata, α0, λ0)
    @test err / norm(ydata) < 1.0e-8
end