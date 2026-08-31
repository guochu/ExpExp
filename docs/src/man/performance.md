# 精度与效率

本文数据来自 `performance/` 下的基准脚本：

- `performance/benchmark.jl`：复数与实数、含噪 / 无噪、变动数据长度 `N` 的完整对比；
- `performance/tol_scaling.jl`：容差缩放行为（积分误差随网格加密是否稳定）。

## 精度（相对重建误差）

基准：`p = 3`，`N = 200`，随加性高斯噪声强度 `σ` 变化。

| 噪声σ | `DeterminedProny` | `OverDeterminedProny` | `MatrixPencil` | `LeastSquareProny` |
|---|---|---|---|---|
| 0（无噪） | 3.2e-16 | 1.8e-16 | 3.7e-16 | 1.1e-16 |
| 1e-6 | **9.4e127（发散）** | 6.9e-6 | 6.9e-6 | 6.8e-6 |
| 1e-4 | 7.5e-4 | 7.5e-4 | 7.2e-4 | 7.2e-4 |
| 1e-2 | 6.9e-2 | 6.9e-2 | 7.0e-2 | 6.9e-2 |

复数指数和（`p = 2`，精确）下四种方法误差均约 `1e-15`（`LeastSquareProny` 为 9.0e-16）。

**要点**：

- **无噪、精确**数据下，四者都接近机器精度；
- **抗噪**：`DeterminedProny` 在 `σ = 1e-6` 时即数值发散（相对误差爆到 10¹²⁷），只适合无噪
  精确数据；其余三者稳健，误差基本等于噪声水平；
- 四者均支持实数与复数数据。

## 效率（墙钟耗时，最佳 5 次之一，`p = 4`，无噪）

| N | `DeterminedProny` | `OverDeterminedProny` | `MatrixPencil` | `LeastSquareProny` |
|---|---|---|---|---|
| 100 | 0.006 ms | 0.034 ms | 0.237 ms | 0.489 ms |
| 500 | 0.006 ms | 0.096 ms | 5.259 ms | 1.179 ms |
| 1000 | 0.006 ms | 0.276 ms | 106.7 ms | 2.928 ms |
| 2000 | 0.006 ms | 0.527 ms | 816.8 ms | 5.460 ms |

**要点**：

- `DeterminedProny` 只用前 `2p` 个点，耗时与 `N` 无关（恒 ~5µs），最快；
- `OverDeterminedProny` 耗时随 `N` 近似线性增长；
- `LeastSquareProny` 在 `OverDeterminedProny` 基础上再做梯度精化，耗时随 `N` 近似线性增长，
  `N = 2000` 时约 5 ms（比 `OverDeterminedProny` 慢约一个量级，仍远快于 `MatrixPencil`）；
- `MatrixPencil` 需对约 `(N/2)²` 的 Hankel 矩阵做 SVD，耗时大致随 `N` 立方增长，
  `N = 2000` 时超过 0.8 s。

## 容差缩放与网格无关性

内部实际容差为

```math
\text{tol}_\text{actual} = \text{alg.tol} \cdot \| f \| .
```

其中 `‖f‖ ≈ c·√N`（`c` 为信号 RMS，网格无关）。这等价于标准**相对 L2 残差**判据
`‖f_pred − f‖ / ‖f‖ ≤ tol`，分子分母同阶，比率网格无关。因此对一个连续函数用不同步长
离散化时，**逐点平均误差与积分误差 `Σ|eᵢ|·h` 不随离散化步长减小而明显变化**。

`performance/tol_scaling.jl` 以 `f(x) = √x`（在 `x = 0` 处导数发散、需要较大项数的函数）
验证了这一性质。`N = 16 → 512`（加密 32 倍）的结果：

| N | `OverDeterminedProny` 积分误差 | `MatrixPencil` 积分误差 | `LeastSquareProny` 积分误差 |
|---|---|---|---|
| 16 | 2.4e-8 | 5.2e-7 | 5.0e-7 |
| 64 | 2.2e-7 | 1.8e-7 | 9.7e-8 |
| 256 | 3.7e-7 | 3.9e-7 | 3.9e-7 |
| 512 | 1.05e-6 | 1.9e-7 | 4.0e-7 |

`MatrixPencil` 与 `LeastSquareProny` 的积分误差在 1e-7–5e-7 间波动，无明显漂移；
`OverDeterminedProny` 在 `N = 512` 时因 `√x` 在 0 点导数奇异性导致 Prony 线性系统病态而略有
退化——这是该算法的真实局限，`LeastSquareProny` 的 LM 精化对此更有韧性。
（`DeterminedProny` 对非指数和函数直接失效，未列入。）

!!! note "为什么不采用 `‖f‖²` 缩放"
    若改用 `tol_actual = alg.tol * ‖f‖²`，允许的逐点误差会随 `√N` 增长，积分误差上界放大
    `√N` 倍，网格越细拟合反而越差，违背上述合理行为，故不推荐。

## 结论与取舍建议

- 追求**抗噪稳健**：`MatrixPencil`；
- 无噪小数据图快：`DeterminedProny`（仅测试用）；
- 含噪场景**速度与精度折中**：`OverDeterminedProny` / `LeastSquareProny`，后者以梯度精化
  获得略高的拟合精度（实数 / 复数均可）；
- 数据过密或基底接近导致条件数差：增大 `stepsize` 或设 `stepsize = nothing` 自动选择。
