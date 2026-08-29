# 接口修改记录

本文档记录 `ExpExp` 包的关键接口变更，便于后续维护时参考。

## 算法与文件结构

每个算法由独立文件实现（与主模块 `ExpExp` 同目录结构对应的方式 include）：

- `src/expansion.jl` —— 公共抽象类型与统一接口（`exponential_expansion` / `exponential_expansion_opt` / `expansion_error`），含 `first_period`、`cut` 工具。
- `src/prony.jl` —— Prony 族底层求解 `determined_prony`、`overdetermined_prony`。
- `src/matrixpencil.jl` —— 矩阵束法 `matrix_pencil`。
- `src/lsgexpansion.jl` —— 最小二乘 + 梯度精化 `LeastSquareProny`（仅复数）。

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

`stepsize` 从算法类型字段中移除，改为最外层关键字参数（默认 1）：

- `exponential_expansion(f, alg; stepsize=1)`
- `exponential_expansion(f, L, alg; stepsize=1)`

算法类型仅保留 `n / tol / verbosity` 字段。

新增自动选步长接口：

- `exponential_expansion_opt(f, alg)` —— 基于 `first_period` 生成候选步长串，逐个拟合取误差最小者，并 `cut` 剪枝。

## 导出调整

- 非导出但保留测试：`lsq_expansion_n`（梯度精化的底层入口），测试中通过 `ExpExp.lsq_expansion_n` 限定调用。

## 已知约束

- `prony`（现 `determined_prony`）对含噪数据数值不稳定，仅适合无噪数据。
- `LeastSquareProny` 仅支持复数据。