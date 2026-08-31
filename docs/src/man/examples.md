# 示例

以下示例假定已经 `using ExpExp`。

## 1. 基本指数和

```@example ex1
using ExpExp

# 真值：f(k) = 0.5^k + 2 * 0.7^k
f = [0.5^k + 2 * 0.7^k for k in 1:20]

xs, lambdas = exponential_expansion(f, OverDeterminedProny(n=20))

# 重建并检查误差
f_pred = [sum(xs[i] * lambdas[i]^k for i in 1:length(xs)) for k in 1:20]
rel_err = expansion_error(f, xs, lambdas) / norm(f)
```

## 2. 复数数据

```@example ex2
using ExpExp

# 含衰减振动的复数指数和
f = [(0.6 + 0.3im) * (0.9 + 0.2im)^k + (0.2 - 0.1im) * (0.85 - 0.15im)^k for k in 1:100]

xs, lambdas = exponential_expansion(f, MatrixPencil(n=10))
expansion_error(f, xs, lambdas) / norm(f)
```

```@example ex2
# 复数初猜 + LM 精化
xs, lambdas = exponential_expansion(f, LeastSquareProny(n=10, tol=1e-10))
expansion_error(f, xs, lambdas) / norm(f)
```

## 3. 以函数形式输入

传入可调用对象与采样点数 `L`，在 `1:L` 上采样后展开：

```@example ex3
using ExpExp

g(k) = 0.5^k + 2 * 0.7^k
xs, lambdas = exponential_expansion(g, 30, OverDeterminedProny())
expansion_error([g(k) for k in 1:30], xs, lambdas) / norm([g(k) for k in 1:30])
```

## 4. 采样步长

```@example ex4
using ExpExp

f = [0.5^k + 2 * 0.7^k for k in 1:30]

# 固定步长：只使用 f[1], f[4], f[7], ... 拟合，再换算回原网格
xs, lambdas = exponential_expansion(f, OverDeterminedProny(stepsize=3))
expansion_error(f, xs, lambdas) / norm(f)
```

```@example ex4
# 自动选步长
xs, lambdas = exponential_expansion(f, OverDeterminedProny(stepsize=nothing))
expansion_error(f, xs, lambdas) / norm(f)
```

## 5. 含噪数据与抗噪对比

```@example ex5
using ExpExp
using Random

Random.seed!(1234)

f_clean = [0.5^k + 2 * 0.7^k for k in 1:200]
σ = 1e-4
f_noisy = f_clean .+ σ .* randn(length(f_clean))

xs1, λ1 = exponential_expansion(f_noisy, OverDeterminedProny(n=6, tol=1e-6))
err_od = expansion_error(f_noisy, xs1, λ1) / norm(f_noisy)    # ~ 1e-4

xs2, λ2 = exponential_expansion(f_noisy, MatrixPencil(n=6, tol=1e-6))
err_mp = expansion_error(f_noisy, xs2, λ2) / norm(f_noisy)    # ~ 1e-4

# 注意：DeterminedProny 对噪声极敏感，含噪时通常发散，仅适合无噪精确数据
(err_od, err_mp)
```

## 6. 验证自动步长的网格无关性

```@example ex6
using ExpExp

# 非指数和函数（例如 f(x) = 1/(1+x²)），以不同格点数离散化
# 合理行为：平均逐点误差（等价于积分误差除以区间长度）不随格点加密而明显变化
function int_err(N)
    xs = (0:N-1) ./ (N-1)
    f = [1 / (1 + x^2) for x in xs]
    xs_fit, λ = exponential_expansion(f, OverDeterminedProny(n=10, tol=1e-6))
    errs = abs.([sum(xs_fit[i] * λ[i]^k for i in 1:length(xs_fit)) for k in 1:N] .- f)
    return sum(errs) / N
end

[int_err(N) for N in (16, 64, 256)]   # 量级应基本稳定
```

## 7. 提取有效项数

自动选步长路径在返回前会调用 [`cut`](@ref) 按系数幅值裁剪冗余项：

```@example ex7
using ExpExp

# 真实只有 2 个指数，但给足 n
f = [0.5^k + 2 * 0.7^k for k in 1:40]
xs, lambdas = exponential_expansion(f, OverDeterminedProny(n=20, stepsize=nothing))
length(xs)   # 收敛项数（通常为 2，其余被剪掉）
```

## 8. 自定义算法接入公共流程

继承 [`ExponentialExpansionAlgorithm`](@ref) 并实现 [`exponential_expansion_n`](@ref)：

```@example ex8
using ExpExp

struct MyAlg <: ExponentialExpansionAlgorithm
    n::Int
    tol::Float64
    verbosity::Int
    stepsize::Union{Int, Nothing}
end

function ExpExp.exponential_expansion_n(f::Vector, p::Int, alg::MyAlg)
    # 返回 (coeffs, bases)，满足 f(k) ≈ Σᵢ coeffs[i] * bases[i]^k
    return overdetermined_prony(f, p)   # 示例：直接复用现成方法
end

f = [0.5^k + 2 * 0.7^k for k in 1:20]
xs, lambdas = exponential_expansion(f, MyAlg(20, 1e-8, 1, 1))
expansion_error(f, xs, lambdas) / norm(f)
```
