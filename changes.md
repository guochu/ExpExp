# 接口修改记录

本文档记录 `ExpExp` 包的关键接口变更，便于后续维护时参考。

## 算法与文件结构

每个算法由独立文件实现（与主模块 `ExpExp` 同目录结构对应的方式 include）：

- `src/expansion.jl` —— 公共抽象类型与统一接口（`exponential_expansion` / `expansion_error`），含 `first_period`、`cut` 工具。
- `src/prony.jl` —— Prony 族底层求解 `determined_prony`、`overdetermined_prony`。
- `src/matrixpencil.jl` —— 矩阵束法 `matrix_pencil`。
- `src/leastsquareprony.jl` —— 最小二乘 + 梯度精化 `LeastSquareProny`（实/复数，实数内部自动升型）。

## 类型改名

| 旧名称 | 新名称 |
| --- | --- |
| `PronyExpansion` | `OverDeterminedProny` |
| `DeterminedPronyExpansion` | `DeterminedProny` |
| `MatrixPencilExpansion` | `MatrixPencil` |
| `LsqExpansion` | `LeastSquareProny` |

抽象基类 `ExponentialExpansionAlgorithm`、`AbstractPronyExpansion`（仅 Prony 族）保持不变。

继承关系：

```
ExponentialExpansionAlgorithm
├── AbstractPronyExpansion
│   ├── OverDeterminedProny
│   └── DeterminedProny
├── MatrixPencil
└── LeastSquareProny
```

## 底层函数改名

| 旧名称 | 新名称 |
| --- | --- |
| `prony` | `determined_prony` |
| `lsq_prony` | `overdetermined_prony` |

## 最外层接口

`stepsize` 作为算法类型的字段 `alg.stepsize`（类型 `Union{Int, Nothing}`，默认 `1`）传入，
算法类型字段为 `n / tol / verbosity / stepsize`：

- `exponential_expansion(f, alg)` —— `alg.stepsize` 为整数 `s`（默认 1）时，在均匀子采样
  `f[1:s:end]` 上拟合并换算回原始采样；
- `exponential_expansion(f, L, alg)` —— 先在 `1:L` 上采样再展开。

`alg.stepsize === nothing` 时自动选步长（原 `exponential_expansion_opt` 接口已移除，并入
`exponential_expansion`）：基于 `first_period` 生成候选步长串，逐个拟合取误差最小者，并 `cut` 剪枝。

## 导出调整

- 移除 `exponential_expansion_opt`，自动选步长改为 `alg.stepsize = nothing` 触发。
- 非导出但保留测试：`lsq_expansion_n`（梯度精化的底层入口），测试中通过 `ExpExp.lsq_expansion_n` 限定调用。

## 已知约束

- `prony`（现 `determined_prony`）对含噪数据数值不稳定，仅适合无噪数据。
- `LeastSquareProny` 支持实/复数数据（实数内部自动升为复数）。