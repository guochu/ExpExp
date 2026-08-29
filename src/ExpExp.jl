module ExpExp

using LinearAlgebra
using Polynomials: Polynomial, roots

export ExponentialExpansionAlgorithm, OverDeterminedProny, DeterminedProny, MatrixPencil, LeastSquareProny
export exponential_expansion, exponential_expansion_opt, expansion_error
export determined_prony, overdetermined_prony, matrix_pencil, leastsquare_prony

include("expansion.jl")
include("prony.jl")
include("matrixpencil.jl")
include("leastsquareprony.jl")

end