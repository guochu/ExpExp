# ==========================================================================
# Precision & efficiency analysis of the sum-of-exponentials fitting algorithms
#
#   Compare:  determined_prony (deterministic), overdetermined_prony (least-squares),
#             matrix_pencil (Hua–Sarkar, SVD-based)
#
#   Run with:
#     JULIA_PROBE_LIBSTDCXX=0 JULIA_PKG_OFFLINE=true \
#     JULIA_DEPOT_PATH=<project>/performance/../.julia_depot \
#     OMP_NUM_THREADS=1 julia --project=. performance/benchmark.jl
# ==========================================================================

using ExpExp
using LinearAlgebra
using Random
using Printf

Random.seed!(42)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
rel_err(y, xs, zs) = expansion_error(y, xs, zs) / norm(y)

# wall-clock: warm up for JIT, then take the minimum of `reps` runs
function best_time(reps::Int, f)
    f()                                   # JIT / first compile
    tmin = Inf
    for _ in 1:reps
        t = @elapsed f()
        tmin = min(tmin, t)
    end
    tmin
end

# model signal with `p` damped exponentials (optional perturbation -> "noise")
function make_data(p::Int, N::Int; noise_std::Float64=0.0,
                   complex_data::Bool=false)
    if complex_data
        bases = [(0.9 - 0.08k) * exp(0.03im * k) for k in 1:p]   # distinct complex |z|<1
        coeff = [(1.0 + 0.5im) * (0.5 + 0.1im)^k for k in 1:p]
        y = zeros(ComplexF64, N)
        noiserm = randn(ComplexF64, N)
    else
        bases = [0.5 - 0.03k for k in 1:p]                       # distinct real |z|<1
        coeff = [1.0 + 0.2k for k in 1:p]
        y = zeros(Float64, N)
        noiserm = randn(N)
    end
    for n in 1:N
        acc = zero(eltype(y))
        for k in 1:p
            acc += coeff[k] * bases[k]^n
        end
        y[n] = acc
    end
    if noise_std > 0
        y .+= noise_std .* noiserm
    end
    y
end

# algorithm -> (table label, display name, fit function)
# leastsquare_prony upcasts real data to ComplexF64 internally.
const ALGS = [
    (label="determined_prony", name="determined_prony (deterministic)",
     fit=(y, p) -> ExpExp.determined_prony(y, p)),
    (label="overdetermined_prony", name="overdetermined_prony (least-squares)",
     fit=(y, p) -> ExpExp.overdetermined_prony(y, p)),
    (label="m_pencil", name="matrix_pencil (Hua-Sarkar)",
     fit=(y, p) -> ExpExp.matrix_pencil(y, p)),
    (label="lsq_prony", name="leastsquare_prony (complex + LM refine)",
     fit=(y, p) -> leastsquare_prony(y, p)),
]

println("="^78)
println(" 1) PRECISION: relative reconstruction error vs. noise level")
println("    signal: p=3 damped 'almost-real' bases, N=200")
println("="^78)

p, N = 3, 200
@printf "%-14s" "noise \\ alg"
for alg in ALGS
    @printf " %14s" alg.label
end
println()
for snr in (0.0, 1e-6, 1e-4, 1e-2)
    y = make_data(p, N; noise_std=snr)
    @printf "%-14.0e" snr
    for alg in ALGS
        xs, zs = alg.fit(y, p)
        @printf " %13.2e" rel_err(y, xs, zs)
    end
    println()
end

println()
println("="^78)
println(" 2) PRECISION on purely complex-exponential data (p=2, exact)")
println("="^78)
yc = make_data(2, 100; complex_data=true)
for alg in ALGS
    xs, zs = alg.fit(yc, 2)
    @printf "  %-28s relative error = %.2e\n" alg.name rel_err(yc, xs, zs)
end

println()
println("="^78)
println(" 3) EFFICIENCY: best-of-5 wall time [ms] vs. data length N (p=4, clean)")
println("="^78)
p = 4
@printf "%-10s" "N"
for alg in ALGS
    @printf " %14s" alg.label
end
println()
for N in (100, 500, 1000, 2000)
    y = make_data(p, N)
    row = Float64[]
    for alg in ALGS
        push!(row, 1e3 * best_time(5, () -> alg.fit(y, p)))
    end
    @printf "%-10d" N
    for t in row
        @printf " %11.3f ms" t
    end
    println()
end

println()
println("Done.")