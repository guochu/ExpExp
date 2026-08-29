module ExpExp

using LinearAlgebra
using Polynomials: Polynomial, roots

export ExponentialExpansionAlgorithm, PronyExpansion, DeterminedPronyExpansion, MatrixPencilExpansion
export exponential_expansion, expansion_error, prony, lsq_prony, matrix_pencil

include("expansion.jl")
include("prony.jl")
include("matrixpencil.jl")

end