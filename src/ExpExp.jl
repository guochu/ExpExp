module ExpExp

using LinearAlgebra
using Polynomials: Polynomial, roots

export ExponentialExpansionAlgorithm, PronyExpansion, DeterminedPronyExpansion, MatrixPencilExpansion
export PronyExpansion2, LsqExpansion2
export exponential_expansion, expansion_error, prony, lsq_prony, matrix_pencil, lsq_expansion_n

include("expansion.jl")
include("prony.jl")
include("matrixpencil.jl")
include("gradientdescent.jl")

end