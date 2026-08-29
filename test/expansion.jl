# ==========================================================================
# Common interface tests: exponential_expansion, expansion_error, stepsize
# ==========================================================================

@testset "exponential_expansion" begin
    atol = 1.0e-10
    for ydata in real_exp_data()
        for alg in (OverDeterminedProny(n=20, tol=atol), DeterminedProny(n=20, tol=atol))
            xs, lambdas = exponential_expansion(ydata, alg)
            @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
        end
    end
end

@testset "exponential_expansion: function input" begin
    ydata = [0.5^k + 2 * 0.7^k for k in 1:20]
    atol = 1.0e-8
    xs, lambdas = exponential_expansion(k -> 0.5^k + 2 * 0.7^k, 20, OverDeterminedProny(n=20, tol=atol))
    @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
    xs, lambdas = exponential_expansion(k -> 0.5^k + 2 * 0.7^k, 20, alg=OverDeterminedProny(n=20, tol=atol))
    @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
end

@testset "exponential_expansion: stepsize" begin
    ydata = [0.5^k + 2 * 0.7^k for k in 1:20]
    stepsize = 3
    atol = 1.0e-8
    xs, lambdas = exponential_expansion(ydata, OverDeterminedProny(n=20, tol=atol, stepsize=stepsize))
    @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
end

@testset "expansion_error: concatenated form" begin
    ydata = [0.5^k for k in 1:10]
    err = expansion_error(ydata, vcat([1.0], [0.5]))
    @test err ≈ 0 atol = 1.0e-12
end

@testset "complex data: common interface" begin
    atol = 1.0e-8
    ydata = complex_exp_data()
    for alg in (OverDeterminedProny(n=10, tol=atol),
                DeterminedProny(n=10, tol=atol),
                MatrixPencil(n=10, tol=atol))
        xs, lambdas = exponential_expansion(ydata, alg)
        @test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
    end
end