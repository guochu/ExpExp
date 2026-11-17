# 接口修改记录

本文档记录 `ExpExp` 包的关键接口变更，便于后续维护时参考。

## 算法与文件结构

每个算法由独立文件实现（与主模块 `ExpExp` 同目录结构对应的方式 include）：

- `src/expansion.jl` —— 公共抽象类型与统一接口（`exponential_expansion` / `expansion_error`），含 `first_period`、`cut` 工具。
- `src/prony.jl` —— Prony 族底层求解 `determined_prony`、`overdetermined_prony`。
- `src/matrixpencil.jl` —— 矩阵束法 `matrix_pencil`。
- `src/leastsquareprony.jl` —— 最小二乘 + 梯度精化 `LeastSquareProny`（实/复数，实数内部自动升型）。
- `src/anchoredpencil.jl` —— 行子采样 + 双端锚定矩阵束 `anchored_matrix_pencil` / `anchored_matrix_pencil_sweep` / `AnchoredMatrixPencil`。

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
├── LeastSquareProny
└── AnchoredMatrixPencil
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

## AnchoredMatrixPencil 新算法（2026-11-16）

GTEMPO `fit_exponentials` 的领域扩展上收为 ExpExp 通用算法（见 `plan/expexp_anchored_pencil_plan.md`）：

- `anchored_matrix_pencil(s, p; pencil_rows=128) -> (xs, lambdas, anchors)`
  —— Hankel 行偏移均匀子采样至 `pencil_rows` 行（铅笔列空间 = 极点 Vandermonde 张成，行选择不变）
  + 双端锚定：返回三元组在拟合窗 `d = 1..L` 上满足 `:left` 项 `xs·λ^(d-1)`、`:right` 项 `xs·λ^(L-d)`，
  全部 `|λ| ≤ 1`（`|q|>1` 极点一律折叠为右锚 `1/q`，恒等式 `q^(d-1) = q^(L-1)·(1/q)^(L-d)`，因子吸收进权重）。
  **无投影分支**：2026-11-17 裁决删除 `rightanchors` 输入（静默投影超临界极点被裁决为错误形态，
  算法恒按双端锚定执行；GTEMPO 实轴调用点先行翻转）。
- `anchored_matrix_pencil_sweep(s; tol=1e-10, n=40, pencil_rows=128)
  -> (xs, lambdas, anchors, err)` —— 一次 H0 SVD 复用的最小阶秩扫：首阶相对重建误差 `err ≤ tol`
  即返回；扫完不达抛 `ArgumentError`（带最优 err）。与逐调 `anchored_matrix_pencil` 逐位等价但只做一次 SVD，
  性能敏感消费走这层。
- `AnchoredMatrixPencil <: ExponentialExpansionAlgorithm`，字段 `n / tol / verbosity / pencil_rows /
  stepsize`，默认值 = GTEMPO 现设置（40 / 1e-10 / 1 / 128 / 1）。`exponential_expansion_n`
  经可逆展平返回包约定 `(xs, λ)`：右锚项 `xs·q^(L-k)` 映射为 `(xs·q^L, 1/q)`（`|λ|>1` ⟺ 右锚，node = `1/λ`），
  展平仅在拟合窗 `k = 1..L` 上值恒等；还原（消费端）：`:left` 权重乘 `λ`、`:right` 权重乘 `λ^L`。
- 结构面走共享循环时每阶重做 Hankel SVD（纯、O(n) 倍代价，docstring 已注明）；`anchored_matrix_pencil_sweep`
  为一次 SVD 版本。
- `AnchoredMatrixPencil` 的 `verbosity` 字段为共享循环（`_exponential_expansion_impl` / `_exponential_expansion_step`）
  的硬要求，补于计划草图之外。