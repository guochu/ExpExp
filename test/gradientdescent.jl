# ==========================================================================
# Gradient-refined expansion tests: PronyExpansion2 / LsqExpansion2
# ==========================================================================

@testset "PronyExpansion2: real data" begin
    atol = 1.0e-8
    for ydata in real_exp_data()
        xs, lambdas = exponential_expansion(ydata, PronyExpansion2(n=10, atol=atol))
        @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
    end
end

@testset "PronyExpansion2: complex data" begin
    atol = 1.0e-8
    ydata = complex_exp_data()
    xs, lambdas = exponential_expansion(ydata, PronyExpansion2(n=10, atol=atol))
    @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
end

@testset "LsqExpansion2: complex data" begin
    atol = 1.0e-8
    ydata = complex_exp_data()
    xs, lambdas = exponential_expansion(ydata, LsqExpansion2(n=10, atol=atol))
    @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
end

@testset "LsqExpansion2: rejects real data" begin
    @test_throws ErrorException exponential_expansion(real_exp_data()[1], LsqExpansion2())
end

@testset "lsq_expansion_n: direct refinement" begin
    # start from an imperfect (slightly perturbed) exact guess and refine it
    ydata = complex_exp_data()
    exact = [(2.0 + 1.0im), (0.5 + 0.3im), (1.0 - 2.0im), (-0.4 + 0.6im)]
    α0 = [exact[1] * (1 + 0.02im); exact[3] * (1 - 0.02im)]
    λ0 = [exact[2] * (1 + 0.03); exact[4] * (1 - 0.03)]
    α, λ, err = lsq_expansion_n(ydata, α0, λ0)
    @test err / norm(ydata) < 1.0e-8
end