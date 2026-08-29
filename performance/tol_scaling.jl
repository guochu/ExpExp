# =============================================================================
# tol 行为测试：连续函数以不同步长离散化时，积分误差是否稳定
#
# exponential_expansion 内部实际使用的收敛容差为
#     tol_actual = alg.tol * norm(f)
# 对固定 tol，norm(f) ~ sqrt(N) * (典型幅值)，即总 L2 误差预算随 sqrt(N) 增长。
#
# 这里检查：网格加密（步长 h -> 0，点数 N -> ∞）时，各算法的
#     err      总 L2 误差（expansion_error）
#     int      积分误差 Σ|e_i|·h ≈ ∫|f - f_pred|dx（比逐点平均更有意义，收敛到连续积分）
#     rms      逐点 RMS（err / sqrt(N)）
# 是否随 N 基本不变（预期：步长不太大时积分误差与离散化无关）。
#
# 注：不测 DeterminedProny（确定性 Prony 对非指数和函数数值不稳定，直接失败/爆炸）。
# =============================================================================

using ExpExp
using LinearAlgebra
using Printf

const ALGS = [
    (label="overdetermined_prony", alg=OverDeterminedProny(n=30, tol=1.0e-6, verbosity=0)),
    (label="matrix_pencil",        alg=MatrixPencil(n=30, tol=1.0e-6, verbosity=0)),
    (label="least_square_prony",   alg=LeastSquareProny(n=30, tol=1.0e-6, verbosity=0)),
]

function safe_fit(ydata, alg, h)
    try
        xs, lambdas = exponential_expansion(ydata, alg)
        err = expansion_error(ydata, xs, lambdas)
        fp = [sum(xs[i] * lambdas[i]^k for i in eachindex(xs)) for k in 1:length(ydata)]
        err_int = sum(abs.(fp .- ydata)) * h    # Σ|e_i|·h ≈ ∫|f - f_pred| dx
        return (err=err, err_int=err_int, n=length(xs), ok=true)
    catch
        return (err=NaN, err_int=NaN, n=-1, ok=false)
    end
end

function per_case(title, f, T, Ns)
    println("\n### ", title)
    hdr = rpad("N", 5) * "  " * rpad("h", 9)
    for a in ALGS
        hdr *= "  |  " * rpad(a.label, 32)
    end
    println(hdr)
    println("-" ^ length(hdr))
    for N in Ns
        h = T / (N - 1)
        ydata = [f(k * h) for k in 0:N-1]     # x_k = k*h
        line = lpad(N, 5) * "  " * @sprintf("%.5f", h)
        for a in ALGS
            r = safe_fit(ydata, a.alg, h)
            if r.ok
                line *= "  |  " * @sprintf("err=%.2e int=%.2e rms=%.2e n=%d",
                                           r.err, r.err_int, r.err / sqrt(N), r.n)
            else
                line *= "  |  " * rpad("FAIL", 32)
            end
        end
        println(line)
    end
end

# (a) 本身是指数和的连续函数（精确可表示，2 项）
per_case("f(x) = exp(0.2x) + 2exp(-0.1x)  on [0, 4]",
         x -> exp(0.2x) + 2.0 * exp(-0.1x), 4.0,
         [16, 32, 64, 128, 256, 512])

# (b) 更复杂的连续函数（非有限指数和）：sqrt(x) 在 x=0 处导数发散，
#     指数近似需要明显更多的项（n 约 8~15），且误差落在 tol 边界附近。
per_case("f(x) = sqrt(x)  on [0, 1]",
         x -> sqrt(x), 1.0,
         [16, 32, 64, 128, 256, 512])
