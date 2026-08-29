module ExpExp

using LinearAlgebra
using Polynomials: Polynomial, roots

export ExponentialExpansionAlgorithm, OverDeterminedProny, DeterminedProny, MatrixPencil, LeastSquareProny
export exponential_expansion, exponential_expansion_opt, expansion_error, determined_prony, overdetermined_prony, matrix_pencil

include("expansion.jl")
include("prony.jl")
include("matrixpencil.jl")
include("lsgexpansion.jl")

end