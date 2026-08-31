# ExpExp.jl

*用指数函数的求和逼近给定的函数或数据序列。*

对数据 `f(1), f(2), …, f(N)`，寻找系数 `αᵢ` 与基底 `λᵢ`，使得

```math
f(k) \approx \sum_{i} \alpha_i \, \lambda_i^k, \qquad k = 1, 2, \ldots, N .
```

同时支持**实数**与**复数**数据与系数。

## 特性

- **统一的抽象接口** [`ExponentialExpansionAlgorithm`](@ref)，四类算法
  （[`OverDeterminedProny`](@ref)、[`DeterminedProny`](@ref)、[`MatrixPencil`](@ref)、
  [`LeastSquareProny`](@ref)）可自由替换；
- **逐阶迭代 + 误差收敛**的公共流程：从 `n = 1` 项开始逐阶增加指数项，直到重建误差满足
  容差（默认 `tol · ‖f‖`，见 [精度与效率](man/performance.md)），或达到最大项数；
- **采样步长控制**：固定步长 `stepsize = s`（降采样拟合后换算回原采样），或
  `stepsize = nothing` 自动从数据首周期推导候选步长并选取误差最小者；
- **实数 / 复数统一**：所有算法均支持 `Vector{Float64}` 与 `Vector{ComplexF64}` 输入；
- 附带**误差度量** [`expansion_error`](@ref) 与底层单阶拟合函数，便于自定义工作流。

## 安装

```julia-repl
julia> using Pkg

# 本地开发（路径按实际情况填写）
julia> Pkg.develop(path="/path/to/ExpExp")

# 或从 git 仓库安装
julia> Pkg.add(url="https://github.com/guochu/ExpExp")
```

要求 Julia `≥ 1.10`，依赖 `LinearAlgebra`、`Logging`、`Polynomials`。

## 快速开始

```@example quickstart
using ExpExp

# 一阶指数和  f(k) = 0.5^k + 2 * 0.7^k
f = [0.5^k + 2 * 0.7^k for k in 1:20]

# 以最小二乘 Prony 展开
xs, lambdas = exponential_expansion(f, OverDeterminedProny(n=20))

# 重建误差（相对 L2 范数）
expansion_error(f, xs, lambdas) / norm(f)
```

换用其它算法同样简单：

```@example quickstart
xs, lambdas = exponential_expansion(f, MatrixPencil(n=20))
expansion_error(f, xs, lambdas) / norm(f)
```

也支持直接以**函数**形式输入并在 `1:L` 上采样：

```@example quickstart
g(k) = 0.5^k + 2 * 0.7^k
xs, lambdas = exponential_expansion(g, 30, OverDeterminedProny())
expansion_error([g(k) for k in 1:30], xs, lambdas) / norm([g(k) for k in 1:30])
```

## 文档导航

```@contents
Pages = [
    "man/algorithms.md",
    "man/examples.md",
    "man/performance.md",
    "api.md",
]
Depth = 2
```

## 运行测试

```bash
julia --project=. test/runtests.jl
```

测试按单元拆分，见 `test/` 下的 `expansion.jl`、`prony.jl`、`matrixpencil.jl`、`lsqexpansion.jl`。
