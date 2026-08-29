using ExpExp
using LinearAlgebra
using Test
using Random

# A complex sum-of-exponentials test signal: f(k) = Σ αᵢ zᵢ^k
#   with complex coefficients αᵢ and complex bases zᵢ (|zᵢ| < 1).
complex_exp_data(L::Int=30) =
    [(2.0 + 1.0im) * (0.5 + 0.3im)^k + (1.0 - 2.0im) * (-0.4 + 0.6im)^k for k in 1:L]

# Real-valued test signals.
real_exp_data() = [
    [0.5^k + 2 * 0.7^k for k in 1:20],
    [-2 * 0.3^k + 0.9^k - 1.5 * 0.5^k for k in 1:20],
]

@testset "ExpExp.jl" begin
    include("expansion.jl")
    include("prony.jl")
    include("matrixpencil.jl")
    include("lsqexpansion.jl")
end