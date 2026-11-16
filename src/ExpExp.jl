module ExpExp

using LinearAlgebra
using Polynomials: Polynomial, roots

export ExponentialExpansionAlgorithm, OverDeterminedProny, DeterminedProny, MatrixPencil, LeastSquareProny
export AnchoredMatrixPencil
export exponential_expansion, expansion_error
export determined_prony, overdetermined_prony, matrix_pencil, leastsquare_prony
export anchored_matrix_pencil, anchored_matrix_pencil_sweep

include("expansion.jl")
include("prony.jl")
include("matrixpencil.jl")
include("leastsquareprony.jl")
include("anchoredpencil.jl")

end