# 算法

## 公共接口

所有算法都继承抽象类型 [`ExponentialExpansionAlgorithm`](@ref)。每个具体算法只需实现一个
**单阶拟合**方法 [`exponential_expansion_n`](@ref)（把数据 `f` 拟合成 `p` 项指数和），
即可接入公共的逐阶迭代流程。

```julia
ExponentialExpansionAlgorithm
├── AbstractPronyExpansion            （多项式型方法，内部抽象层）
│   ├── OverDeterminedProny            最小二乘 Prony 法
│   └── DeterminedProny                确定性 Hankel Prony 法
├── MatrixPencil                      矩阵束法（Hua–Sarkar，SVD 提取极点）
└── LeastSquareProny                  OverDeterminedProny 初猜 + 梯度下降精化
```

## 逐阶迭代流程

[`exponential_expansion`](@ref) 内部按以下流程工作：

1. 校验 `length(f) > 1`；
2. 计算**实际容差** `tol_actual = alg.tol * norm(f)`（相对 L2 容差，详见
   [精度与效率](man/performance.md)）；
3. 最大迭代项数 `nitr = min(alg.n, div(L, 2))`；
4. 从 `n = 1` 开始，每轮调用 [`exponential_expansion_n`](@ref) 做 `n` 项拟合，计算重建误差
   `err = expansion_error(f, xs, lambdas)`：
   - 若 `err ≤ tol_actual`：收敛，返回 `(xs, lambdas)`；
   - 若 `n` 已到上限附近仍不收敛：发出 `@warn` 并返回当前最好的拟合（不抛异常）。

`verbosity` 控制输出：

| `verbosity` | 行为 |
|---|---|
| `0` | 静默 |
| `1`（默认） | 未收敛时给出 `@warn` |
| `2` | 另打印收敛信息 `converged in n iterations, error is ...` |
| `3` | 另打印系数与基底 |

## 采样步长 `stepsize`

采样步长是算法类型自带的字段 `alg.stepsize`，类型为 `Union{Int, Nothing}`，默认 `1`。

- **`stepsize == 1`（默认）**：直接对原始序列 `f(1), f(2), …, f(N)` 拟合；
- **`stepsize = s > 1`**：先按均匀间隔抽样 `g(n) = f(1 + (n-1)s)`，在降采样序列 `g` 上完成
  拟合后，再通过 [`expansion_changestepsize!`](@ref) 把系数与基底换算回原始采样。因此
  **返回值始终满足** `f(k) ≈ Σᵢ αᵢ λᵢᵏ`（在原始网格上）。换算公式（`α = 1/s`）：

  ```math
  \tilde{\alpha}_i = \alpha_i \, \lambda_i^{1 - 1/s}, \qquad
  \tilde{\lambda}_i = \lambda_i^{1/s} .
  ```

- **`stepsize === nothing`**：自动选择步长。算法基于数据**首周期**（[`first_period`](@ref)）
  推导一组候选步长 `round(period · (0.2, 0.3, 0.35, 0.4, 0.45))` 加上 `1`，逐个尝试，保留
  重建误差最小者，最后再调用 [`cut`](@ref) 按幅值裁剪冗余项。

### 步长的作用与取舍

- 当数据相对真实指数模式**过密**、或各基底彼此接近导致拟合矩阵条件数差时，增大 `stepsize`
  能改善数值稳定性；
- 但步长过大只利用少量采样点，可能丢失高频成分；
- 不确定选多大时，直接设 `stepsize = nothing` 交给算法自动选择。

```@example stepsize
using ExpExp

f = [0.5^k + 2 * 0.7^k for k in 1:30]

# 固定步长：只使用 f[1], f[4], f[7], ... 拟合，再换算回原网格
xs, lambdas = exponential_expansion(f, OverDeterminedProny(stepsize=3))
expansion_error(f, xs, lambdas) / norm(f)

# 自动选步长：基于首周期尝试多个候选步长，取误差最小者
xs, lambdas = exponential_expansion(f, OverDeterminedProny(stepsize=nothing))
expansion_error(f, xs, lambdas) / norm(f)
```

## 四种算法的原理

### `DeterminedProny`（确定性 Prony）

用**精确 Hankel 方法**：只取前 `2p` 个数据点构造 `p × p` Hankel 矩阵，解线性方程组得到
特征多项式系数，求根得基底 `λᵢ`，再解一个 `p × p` 线性系统得系数 `αᵢ`。

!!! warning "仅作测试用"
    `DeterminedProny` 数值上极不稳定、对噪声极其敏感（σ = 1e-6 时即发散到 ~10¹²⁷），对
    非指数和形式的函数也会在加密网格时失效。**仅作为简单测试用例使用**，实际问题请勿选用。

### `OverDeterminedProny`（最小二乘 Prony）

与确定性版本的区别在于构造的是 `(N-p) × p` 的**超定** Hankel 矩阵，用最小二乘求特征
多项式系数；求根得到基底后，再以最小二乘解出系数。用了全部数据，对噪声和项数误差的
鲁棒性显著优于 `DeterminedProny`，是多项式型方法中的推荐选择。

### `MatrixPencil`（矩阵束法，Hua–Sarkar）

不做多项式求根，而是把数据排成 `R × L` Hankel 矩阵（`L ≈ N/2`），对其做 **SVD**，取前
`p` 个左奇异向量去掉首/末行得 `U₁, U₂`，则极点 `λᵢ` 就是 `U₁ \ U₂` 的广义特征值；最后用
最小二乘恢复残差系数。对加性噪声的鲁棒性在四者中最强，代价是需要对 `~ (N/2)²` 的矩阵做
SVD，耗时大致随 `N` 立方增长。

### `LeastSquareProny`（最小二乘 Prony + 梯度精化）

先以 [`overdetermined_prony`](@ref) 得到初猜，再把它重参数化为**实数**未知量
`[|α|; ∠α; |λ|; ∠λ]`，用**阻尼 Gauss–Newton（Levenberg–Marquardt）**最小化实残差
`[Re f; Im f]` 以精化拟合。实数数据内部自动升为 `ComplexF64`。

- 拟合精度通常略高于 `OverDeterminedProny`；
- 单阶拟合需要迭代求解，耗时随 `N` 近似线性增长，`N=2000` 时约 5 ms（比
  `OverDeterminedProny` 慢约一个量级，但远快于 `MatrixPencil`）。

## 输入与输出约定

- **输入**：`Vector{<:Number}`，实数组 `Vector{Float64}` 与复数组 `Vector{ComplexF64}` 均可；
  也支持传入可调用对象 `f` 加采样点数 `L`（在 `1:L` 上采样）。
- **返回**：二元组 `(xs, lambdas)`，满足 `f(k) ≈ Σᵢ xs[i] * lambdas[i]^k`（`k = 1, 2, …`）。
- 系数 `xs` 与基底 `lambdas` 为复数（实数输入时基底通常也落在实轴上，但仍以复数类型返回）。

## 选择指南

| 场景 | 推荐算法 |
|---|---|
| 无噪、项数精确、追求速度 | `DeterminedProny`（仅测试用）或 `OverDeterminedProny` |
| 含噪数据、追求稳健 | `MatrixPencil` |
| 含噪数据、速度与精度折中 | `OverDeterminedProny` / `LeastSquareProny` |
| 追求最高拟合精度 | `LeastSquareProny` |
| 数据过密 / 基底接近 | 增大 `stepsize` 或设 `stepsize = nothing` 自动选择 |
