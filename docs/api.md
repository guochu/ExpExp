# API 参考

## 导出符号

`ExpExp` 导出以下符号：

```
ExponentialExpansionAlgorithm, OverDeterminedProny, DeterminedProny,
MatrixPencil, LeastSquareProny,
exponential_expansion, expansion_error,
determined_prony, overdetermined_prony, matrix_pencil, leastsquare_prony
```

## 抽象类型

### `ExponentialExpansionAlgorithm`

所有展开算法的抽象基类。自定义算法时继承它并实现
`exponential_expansion_n(f, p, alg)` 即可接入公共流程。

### `AbstractPronyExpansion <: ExponentialExpansionAlgorithm`

Prony 系方法的中间抽象层（未导出），`OverDeterminedProny` 与 `DeterminedProny` 继承之。

## 算法类型

四类算法的结构体字段完全相同：

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `n` | `Int` | `10` | 最大指数项数 |
| `tol` | `Float64` | `1e-8` | 收敛相对容差（实际容差为 `tol * norm(f)`） |
| `verbosity` | `Int` | `1` | 输出详细程度（0–3） |
| `stepsize` | `Union{Int, Nothing}` | `1` | 采样步长；`nothing` 表示自动选择 |

均支持位置与关键字两种构造方式，例如
`OverDeterminedProny(20, 1e-8, 1, 1)` 与 `OverDeterminedProny(n=20, tol=1e-8)` 等价。

### `OverDeterminedProny`

最小二乘 Prony 展开算法。`struct OverDeterminedProny <: AbstractPronyExpansion`。

```julia
OverDeterminedProny(n=10, tol=1e-8, verbosity=1, stepsize=1)
OverDeterminedProny(; n=10, tol=1e-8, verbosity=1, stepsize=1)
```

### `DeterminedProny`

确定性 Hankel Prony 展开算法。`struct DeterminedProny <: AbstractPronyExpansion`。

!!! note
    仅作简单测试用：实际数值不稳定且对噪声敏感，建议优先用
    `OverDeterminedProny`、`MatrixPencil` 或 `LeastSquareProny`。

```julia
DeterminedProny(n=10, tol=1e-8, verbosity=1, stepsize=1)
DeterminedProny(; n=10, tol=1e-8, verbosity=1, stepsize=1)
```

### `MatrixPencil`

矩阵束法（Hua–Sarkar）展开算法。`struct MatrixPencil <: ExponentialExpansionAlgorithm`。

```julia
MatrixPencil(n=10, tol=1e-8, verbosity=1, stepsize=1)
MatrixPencil(; n=10, tol=1e-8, verbosity=1, stepsize=1)
```

### `LeastSquareProny`

最小二乘 Prony 初猜 + 阻尼 Gauss–Newton 精化的展开算法。
`struct LeastSquareProny <: ExponentialExpansionAlgorithm`，支持实数 / 复数输入
（实数数据内部自动升为 `ComplexF64`）。

```julia
LeastSquareProny(n=10, tol=1e-8, verbosity=1, stepsize=1)
LeastSquareProny(; n=10, tol=1e-8, verbosity=1, stepsize=1)
```

## 主要入口函数

### `exponential_expansion(f, alg)`

```julia
exponential_expansion(f::Vector{<:Number}, alg::ExponentialExpansionAlgorithm)
```

对数据序列 `f` 做指数展开，返回 `(xs, lambdas)`，满足
`f(k) ≈ Σᵢ xs[i] * lambdas[i]^k`（`k = 1, 2, …, N`）。

步长行为由 `alg.stepsize` 决定：

- `Int`（默认 1）：按固定步长降采样拟合后换算回原采样；
- `nothing`：自动选择步长（基于首周期尝试多个候选步长，取误差最小者并裁剪）。

**其它方法（便捷形式）**：

```julia
exponential_expansion(f::Vector{<:Number}; alg::ExponentialExpansionAlgorithm=OverDeterminedProny())
exponential_expansion(f, L::Int, alg::ExponentialExpansionAlgorithm)
exponential_expansion(f, L::Int; alg::ExponentialExpansionAlgorithm=OverDeterminedProny())
```

其中后两种接受**可调用对象** `f`，先在 `1:L` 上采样再展开。

**错误与警告**：

- `length(f) ≤ 1` 时抛 `ArgumentError("length of data should be larger than 1")`；
- 迭代达到项数上限仍不收敛时发出 `@warn` 并返回当前拟合（`verbosity ≥ 1` 时可见）。

### `expansion_error(f, coeffs, alphas)`

```julia
expansion_error(f::Vector{<:Number}, coeffs::Vector{<:Number}, alphas::Vector{<:Number})
expansion_error(f::Vector{<:Number}, p::Vector{<:Number})
```

计算重建序列与真实序列 `f` 的 **2-范数误差** `‖f_pred - f‖`。

- `expansion_error(f, coeffs, alphas)`：系数与基底分开传入；
- `expansion_error(f, p)`：等价于 `expansion_error(f, vcat(coeffs, alphas))`，
  其中 `p[1:n]` 为系数、`p[n+1:2n]` 为基底。

### `determined_prony(x, p)`

```julia
determined_prony(x::Vector, p::Int)
```

精确 Prony 拟合：把 `x` 拟合成 `p` 项指数和，返回 `(α, z)`，满足
`x(k) ≈ Σᵢ αᵢ * zᵢ^k`。要求 `p ≤ length(x) ÷ 2`。

### `overdetermined_prony(x, p)`

```julia
overdetermined_prony(x::Vector, p::Int)
```

最小二乘 Prony 拟合，返回 `(α, z)`。比 `determined_prony` 对噪声更稳健。

### `matrix_pencil(s, p)`

```julia
matrix_pencil(s::Vector{<:Number}, p::Int)
```

矩阵束法拟合，返回 `(xs, lambdas)`，满足 `s(k) ≈ Σᵢ xs[i] * lambdas[i]^k`。
要求 `1 ≤ p < length(s)`。极点取自 `U₁ \ U₂` 的特征值（`U₁/U₂` 为 Hankel 矩阵 SVD
左奇异向量去掉首/末行），系数由最小二乘恢复。

### `leastsquare_prony(x, p)`

```julia
leastsquare_prony(x::Vector{<:Number}, p::Int)
```

以 `overdetermined_prony` 为初猜、再经阻尼 Gauss–Newton 精化的拟合，返回 `(α, z)`。
实数数据自动升为 `ComplexF64`。

## 内部函数（不保证稳定，供进阶使用）

| 函数 | 说明 |
|---|---|
| `exponential_expansion_n(f, p, alg)` | 单阶拟合接口，每类算法各自的实现 |
| `expansion_changestepsize!(xs, lambdas, stepsize)` | 把降采样拟合结果换算回原始采样 |
| `cut(f, as, bs, atol)` | 按系数幅值从大到小裁剪，保持误差低于 `atol` |
| `lsq_expansion_n(f, alphas, lambdas)` | 对给定初猜做 `(模, 相位)` 参数化的 LM 精化，返回 `(α, λ, rmse)` |
| `first_period(x)` / `_candidate_steps(f)` | 自动步长选择辅助函数 |
