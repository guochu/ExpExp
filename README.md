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
│   ├── PronyExpansion            最小二乘 Prony 法
│   └── DeterminedPronyExpansion   确定性 Hankel Prony 法
└── MatrixPencilExpansion         矩阵束法（Hua–Sarkar，SVD 提取极点）

ExponentialExpansionAlgorithm2（梯度精化族）
├── PronyExpansion2               多降采样步的确定性拟合
└── LsqExpansion2                  lsq_prony 初猜 + 梯度下降精化（复数）
```

| 类型 | 原理 | 数据要求 |
|---|---|---|
| `PronyExpansion` | 最小二乘 Prony（构造 Hankel 矩阵求特征多项式根） | 实数 / 复数 |
| `DeterminedPronyExpansion` | 精确 Prony（仅用前 2p 个点，`prony`） | 实数 / 复数 |
| `MatrixPencilExpansion` | 矩阵束法（Hankel 矩阵 SVD + 广义特征值，`matrix_pencil`） | 实数 / 复数 |
| `PronyExpansion2` | 多步长降采样，取误差最小的 lsq_prony 结果 | 实数 / 复数 |
| `LsqExpansion2` | lsq_prony 初猜，再以 `(模,相位)` 实参数化做梯度下降（`lsq_expansion_n`） | 复数 |

底层可直接调用的求解函数：`prony`、`lsq_prony`、`matrix_pencil`、`lsq_expansion_n`。

---

## 快速开始

```julia
using ExpExp

# 一阶指数和  f(k) = 0.5^k + 2 * 0.7^k
f = [0.5^k + 2 * 0.7^k for k in 1:20]

# 以最小二乘 Prony 展开
xs, lambdas = exponential_expansion(f, PronyExpansion(n=20))
# 或其它算法
xs, lambdas = exponential_expansion(f, MatrixPencilExpansion(n=20))
xs, lambdas = exponential_expansion(f, LsqExpansion2(atol=1e-8))   # 复数数据

# 重建误差（相对 L2 范数）
rel_err = expansion_error(f, xs, lambdas) / norm(f)
```

也支持直接以**函数**形式输入并在 `1:L` 上采样：

```julia
xs, lambdas = exponential_expansion(k -> 0.5^k + 2 * 0.7^k, 20, PronyExpansion())

# 列名形式等价
xs, lambdas = exponential_expansion(k -> 0.5^k + 2 * 0.7^k, 20, alg=PronyExpansion())
```

设置步长：

```julia
xs, lambdas = exponential_expansion(f[1:3:end], PronyExpansion(stepsize=3))
```

---

## 运行测试

```bash
julia --project=. test/runtests.jl
```

测试按单元拆分，见 `test/` 目录下的 `expansion.jl`、`prony.jl`、`matrixpencil.jl`、
`gradientdescent.jl`。

---

## 精度与效率分析

结论来自 `performance/benchmark.jl`（复数与实数、含噪/无噪、变动数据长度 `N`）。

### 精度（相对重建误差，p=3，N=200，随噪声强度变化）

| 噪声σ | prony（确定性） | lsq_prony | matrix_pencil |
|---|---|---|---|
| 0（无噪） | 3.2e-16 | 1.8e-16 | 3.7e-16 |
| 1e-6 | **9.4e127（发散）** | 6.9e-6 | 6.9e-6 |
| 1e-4 | 7.5e-4 | 7.5e-4 | 7.2e-4 |
| 1e-2 | 6.9e-2 | 6.9e-2 | 7.0e-2 |

复数指数和（p=2，精确）下三种方法误差均约 `1e-15`，均**支持复数**。

### 效率（最佳 5 次之一的墙钟耗时，p=4，无噪）

| N | prony | lsq_prony | matrix_pencil |
|---|---|---|---|
| 100 | 0.005 ms | 0.013 ms | 0.122 ms |
| 500 | 0.005 ms | 0.079 ms | 4.94 ms |
| 1000 | 0.005 ms | 0.260 ms | 88.9 ms |
| 2000 | 0.005 ms | 0.435 ms | 645.8 ms |

### 结论

- **无噪、复数、精确**数据下，`prony`、`lsq_prony`、`matrix_pencil` 都接近机器精度。
- **抗噪**：确定性 `prony` 在 σ=1e-6 时即数值发散（相对误差爆到 10¹²⁷），只适合无噪声精确数据；
  `lsq_prony` 与 `matrix_pencil` 稳健，误差基本等于噪声水平。
- **效率**：
  - `prony` 只用到前 2p 个点，耗时与 `N` 无关（恒 ~5µs），最快；
  - `lsq_prony` 随 `N` 近似线性增长；
  - `matrix_pencil` 需对约 `(N/2)²` 的 Hankel 矩阵做 SVD，大致随 `N` 立方增长，`N=2000` 时超过 0.6 s。
- **取舍建议**：追求抗噪稳健用 `matrix_pencil`；无噪小数据图快用 `prony`；
  含噪场景下 `lsq_prony` 在强度与速度间折中最佳；对复数序列可进一步用
  `LsqExpansion2` 以梯度精化获得更高的拟合精度。