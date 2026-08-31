# API 参考

```@meta
CurrentModule = ExpExp
```

`ExpExp` 导出的符号：

```julia
ExponentialExpansionAlgorithm, OverDeterminedProny, DeterminedProny,
MatrixPencil, LeastSquareProny,
exponential_expansion, expansion_error,
determined_prony, overdetermined_prony, matrix_pencil, leastsquare_prony
```

## 抽象类型

```@docs
ExponentialExpansionAlgorithm
AbstractPronyExpansion
```

## 算法类型

四类算法的结构体字段完全相同：

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `n` | `Int` | `10` | 最大指数项数 |
| `tol` | `Float64` | `1e-8` | 收敛相对容差（实际容差为 `tol * norm(f)`） |
| `verbosity` | `Int` | `1` | 输出详细程度（0–3） |
| `stepsize` | `Union{Int, Nothing}` | `1` | 采样步长；`nothing` 表示自动选择 |

均支持位置与关键字两种构造方式，例如 `OverDeterminedProny(20, 1e-8, 1, 1)` 与
`OverDeterminedProny(n=20, tol=1e-8)` 等价。

```@docs
OverDeterminedProny
DeterminedProny
MatrixPencil
LeastSquareProny
```

## 主要入口函数

```@docs
exponential_expansion
expansion_error
```

## 底层求解函数

这些函数直接对数据做**单阶 `p` 项拟合**，不经过逐阶迭代与步长处理；返回约定与
`exponential_expansion` 一致：`(α, z)` 满足 `x(k) ≈ Σᵢ αᵢ * zᵢ^k`。

```@docs
determined_prony
overdetermined_prony
matrix_pencil
leastsquare_prony
```

## 内部函数（不保证稳定，供进阶使用）

```@docs
exponential_expansion_n
expansion_changestepsize!
cut
lsq_expansion_n
first_period
```
