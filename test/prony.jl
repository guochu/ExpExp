# ==========================================================================
# Prony algorithm tests: determined_prony / overdetermined_prony (real and complex data)
# ==========================================================================

@testset "Prony: deterministic on real data" begin
    atol = 1.0e-10
    # (data, number of exponential terms)
    cases = [
        ([0.5^k + 2 * 0.7^k for k in 1:20], 2),
        ([-2 * 0.3^k + 0.9^k - 1.5 * 0.5^k for k in 1:20], 3),
    ]
    for (ydata, p) in cases
        α, z = ExpExp.determined_prony(ydata, p)
        @test expansion_error(ydata, α, z) / norm(ydata) < atol
    end
end

@testset "Prony: least-squares on real data" begin
    atol = 1.0e-10
    cases = [
        ([0.5^k + 2 * 0.7^k for k in 1:20], 2),
        ([-2 * 0.3^k + 0.9^k - 1.5 * 0.5^k for k in 1:20], 3),
    ]
    for (ydata, p) in cases
        α, z = ExpExp.overdetermined_prony(ydata, p)
        @test expansion_error(ydata, α, z) / norm(ydata) < atol
    end
end

@testset "Prony: complex data" begin
    ydata = complex_exp_data()
    atol = 1.0e-8
    α, z = ExpExp.determined_prony(ydata, 2)
    @test expansion_error(ydata, α, z) / norm(ydata) < atol
    α, z = ExpExp.overdetermined_prony(ydata, 2)
    @test expansion_error(ydata, α, z) / norm(ydata) < atol
end