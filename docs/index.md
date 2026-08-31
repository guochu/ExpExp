# ExpExp 文档

ExpExp 是一个 Julia 包，用**指数函数的求和**来逼近给定的函数或数据序列：对数据
`f(1), f(2), …, f(N)`，寻找系数 `αᵢ` 与基底 `λᵢ`，使得

```
f(k) ≈ Σᵢ αᵢ · λᵢᵏ,    k = 1, 2, …, N
```

同时支持**实数**与**复数**数据与系数，并提供四种可插拔的算法实现。

## 特性

- **统一的抽象接口** `ExponentialExpansionAlgorithm`，四类算法（`OverDeterminedProny`、
  `DeterminedProny`、`MatrixPencil`、`LeastSquareProny`）可自由替换；
- **逐阶迭代 + 误差收敛**的公共流程：从 `n = 1` 项开始逐阶增加指数项，直到重建误差
  满足容差（默认 `tol · ‖f‖`），或达到最大项数 `n`；
- **采样步长控制**：固定步长 `stepsize = s`（降采样后拟合再换算回原采样），或
  `stepsize = nothing` 自动从数据首周期推导候选步长并选取误差最小者；
- **实数 / 复数统一**：所有算法均支持 `Vector{Float64}` 与 `Vector{ComplexF64}` 输入；
- 附带**误差度量** `expansion_error` 与底层单阶拟合函数，便于自定义工作流。

## 安装

```julia
# 本地开发（路径按实际情况填写）
julia> using Pkg
julia> Pkg.develop(path="/path/to/ExpExp")

# 或从 git 仓库安装
julia> Pkg.add(url="https://github.com/<user>/ExpExp.jl")
```

要求：Julia `≥ 1.10`，依赖 `LinearAlgebra`、`Logging`、`Polynomials`。

## 快速开始

```julia
using ExpExp

# 一阶指数和  f(k) = 0.5^k + 2 * 0.7^k
f = [0.5^k + 2 * 0.7^k for k in 1:20]

# 以最小二乘 Prony 展开
xs, lambdas = exponential_expansion(f, OverDeterminedProny(n=20))

# 或其它算法
xs, lambdas = exponential_expansion(f, MatrixPencil(n=20))
xs, lambdas = exponential_expansion(f, LeastSquareProny(n=20, tol=1e-8))

# 自动选择最优采样步长
xs, lambdas = exponential_expansion(f, OverDeterminedProny(n=20, stepsize=nothing))

# 重建误差（相对 L2 范数）
rel_err = expansion_error(f, xs, lambdas) / norm(f)
```

也支持直接传入**函数**并在 `1:L` 上采样：

```julia
xs, lambdas = exponential_expansion(k -> 0.5^k + 2 * 0.7^k, 20, OverDeterminedProny())
xs, lambdas = exponential_expansion(k -> 0.5^k + 2 * 0.7^k, 20, alg=OverDeterminedProny())
```

## 文档导航

| 章节 | 内容 |
|---|---|
| [算法](algorithms.md) | 公共流程、步长机制、四种算法的原理与数值行为、选择指南 |
| [API 参考](api.md) | 全部公开类型、函数、关键字参数与返回约定 |
| [示例](examples.md) | 覆盖常见场景的可运行示例 |
| [精度与效率](performance.md) | 基准测试结论与容差缩放分析 |

## 运行测试

```bash
julia --project=. test/runtests.jl
```

测试按单元拆分，见 `test/` 下的 `expansion.jl`、`prony.jl`、`matrixpencil.jl`、`lsqexpansion.jl`。
