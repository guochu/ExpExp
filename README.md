# ExpExp

用**指数函数的求和**来逼近给定的函数（序列）：对于数据 `f(1..N)`，寻找系数 `αᵢ` 与基底 `λᵢ`，使得

```
f(k) ≈ Σᵢ αᵢ * λᵢᵏ,    k = 1, 2, ..., N
```

同时支持**实数**与**复数**数据与系数。

---

## 算法

统一的抽象接口为 `ExponentialExpansionAlgorithm`。每个算法通过各自的单阶拟合
（`exponential_expansion_n`）接入「逐阶迭代 + 误差收敛」的公共流程。

```
ExponentialExpansionAlgorithm
├── AbstractPronyExpansion
│   ├── OverDeterminedProny            最小二乘 Prony 法
│   └── DeterminedProny   确定性 Hankel Prony 法
├── MatrixPencil         矩阵束法（Hua–Sarkar，SVD 提取极点）
└── LeastSquareProny                   OverDeterminedProny 初猜 + 梯度下降精化
```

所有算法的采样步长统一通过最外层接口的关键字 `stepsize`（默认 1）传入；需要自动选取最优步长时使用 [`exponential_expansion_opt`](@ref)。

| 类型 | 原理 | 数据要求 |
|---|---|---|
| `OverDeterminedProny` | 最小二乘 Prony（构造 Hankel 矩阵求特征多项式根） | 实数 / 复数 |
| `DeterminedProny` | 精确 Prony（仅用前 2p 个点） | 实数 / 复数 |
| `MatrixPencil` | 矩阵束法（Hankel 矩阵 SVD + 广义特征值） | 实数 / 复数 |
| `LeastSquareProny` | OverDeterminedProny 初猜，再以 `(模,相位)` 实参数化做梯度下降（`lsq_expansion_n`） | 实数 / 复数 |

统一的公开入口为 `exponential_expansion` / `exponential_expansion_opt`，请勿直接调用底层求解函数。

---

## 快速开始

```julia
using ExpExp

# 一阶指数和  f(k) = 0.5^k + 2 * 0.7^k
f = [0.5^k + 2 * 0.7^k for k in 1:20]

# 以最小二乘 Prony 展开
xs, lambdas = exponential_expansion(f, OverDeterminedProny(n=20))
# 或其它算法
xs, lambdas = exponential_expansion(f, MatrixPencil(n=20))
xs, lambdas = exponential_expansion(f, LeastSquareProny(n=20, tol=1e-8))   # 复数数据
# 自动选择最优采样步长
xs, lambdas = exponential_expansion_opt(f, OverDeterminedProny(n=20))

# 重建误差（相对 L2 范数）
rel_err = expansion_error(f, xs, lambdas) / norm(f)
```

也支持直接以**函数**形式输入并在 `1:L` 上采样：

```julia
xs, lambdas = exponential_expansion(k -> 0.5^k + 2 * 0.7^k, 20, OverDeterminedProny())

# 列名形式等价
xs, lambdas = exponential_expansion(k -> 0.5^k + 2 * 0.7^k, 20, alg=OverDeterminedProny())
```

### 采样步长 `stepsize`

默认（`stepsize=1`）时算法直接对原始序列 `f(1), f(2), …, f(N)` 拟合
`f(k) ≈ Σᵢ αᵢ λᵢᵏ`。传入整数 `stepsize=s>1` 时，先按均匀间隔抽样
`g(n) = f(1 + (n-1)s)`，在降采样序列 `g` 上完成拟合后，再把系数与基底换算回原始采样，
因此**返回值始终满足** `f(k) ≈ Σᵢ αᵢ λᵢᵏ`（在原始网格上）：

```julia
# 用步长 3 拟合：内部只用到 f[1], f[4], f[7], ...
xs, lambdas = exponential_expansion(f, OverDeterminedProny(); stepsize=3)
```

步长的作用与取舍：

- 当数据相对真实指数模式**过密**、或各基底彼此接近导致拟合矩阵条件数差时，增大 `stepsize`
  能改善数值稳定性；
- 但步长过大只利用少量采样点，可能丢失高频成分；
- 不确定选多大时，用 `exponential_expansion_opt`：它会基于数据首周期自动尝试多个候选步长，
  返回重建误差最小者。

---

## 运行测试

```bash
julia --project=. test/runtests.jl
```

测试按单元拆分，见 `test/` 目录下的 `expansion.jl`、`prony.jl`、`matrixpencil.jl`、
`lsqexpansion.jl`。

---

## 精度与效率分析

结论来自 `performance/benchmark.jl`（复数与实数、含噪/无噪、变动数据长度 `N`）。

### 精度（相对重建误差，p=3，N=200，随噪声强度变化）

| 噪声σ | DeterminedProny（确定性） | OverDeterminedProny | MatrixPencil | LeastSquareProny |
|---|---|---|---|---|
| 0（无噪） | 3.2e-16 | 1.8e-16 | 3.7e-16 | 1.1e-16 |
| 1e-6 | **9.4e127（发散）** | 6.9e-6 | 6.9e-6 | 6.8e-6 |
| 1e-4 | 7.5e-4 | 7.5e-4 | 7.2e-4 | 7.2e-4 |
| 1e-2 | 6.9e-2 | 6.9e-2 | 7.0e-2 | 6.9e-2 |

复数指数和（p=2，精确）下四种方法误差均约 `1e-15`（LeastSquareProny 为 9.0e-16），均**支持实数与复数**。

### 效率（最佳 5 次之一的墙钟耗时，p=4，无噪）

| N | DeterminedProny | OverDeterminedProny | MatrixPencil | LeastSquareProny |
|---|---|---|---|---|
| 100 | 0.006 ms | 0.034 ms | 0.237 ms | 0.489 ms |
| 500 | 0.006 ms | 0.096 ms | 5.259 ms | 1.179 ms |
| 1000 | 0.006 ms | 0.276 ms | 106.7 ms | 2.928 ms |
| 2000 | 0.006 ms | 0.527 ms | 816.8 ms | 5.460 ms |

### 结论

- **无噪、复数、精确**数据下，`DeterminedProny`、`OverDeterminedProny`、`MatrixPencil`、`LeastSquareProny` 都接近机器精度。
- **抗噪**：确定性 `DeterminedProny` 在 σ=1e-6 时即数值发散（相对误差爆到 10¹²⁷），只适合无噪声精确数据；
  其余三者稳健，误差基本等于噪声水平。
- **效率**：
  - `DeterminedProny` 只用到前 2p 个点，耗时与 `N` 无关（恒 ~5µs），最快；
  - `OverDeterminedProny` 随 `N` 近似线性增长；
  - `LeastSquareProny` 在 `OverDeterminedProny` 基础上再做梯度精化，耗时随 `N` 近似线性增长，
    `N=2000` 时约 5 ms（比 `OverDeterminedProny` 慢约一个量级，仍远快于 `MatrixPencil`）；
  - `MatrixPencil` 需对约 `(N/2)²` 的 Hankel 矩阵做 SVD，大致随 `N` 立方增长，`N=2000` 时超过 0.8 s。
- **取舍建议**：追求抗噪稳健用 `MatrixPencil`；无噪小数据图快用 `DeterminedProny`；
  含噪场景下 `OverDeterminedProny` / `LeastSquareProny` 在强度与速度间折中最佳，后者以梯度精化
  获得略高的拟合精度（实数 / 复数均可）。