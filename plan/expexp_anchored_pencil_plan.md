# ExpExp 新算法 AnchoredMatrixPencil 计划：GTEMPO fit_exponentials 上收为 ExpExp 通用算法

- 日期：2026-11-16
- 基线：ExpExp（本地包）+ GTEMPO HEAD `c01f446c` + 工作区
- 关联：`kernel_and_fit_simplification_plan.md` Part 2 案 B 的落地细化（用户 2026-11-16 翻案采纳：把双端锚定铅笔 + 行子采样 + 秩扫实现为 ExpExp 新算法）；Part 1（kernel corr 视图）独立不受影响

## 0. 已裁决

- 把 GTEMPO `fit_exponentials` 的三件领域扩展（双端锚定 `rightanchors`、行子采样 `pencil_rows`、rtol 秩扫描）实现为 **ExpExp 新算法 `AnchoredMatrixPencil`**，GTEMPO 改为委托调用。
- **默认参数与 GTEMPO 现设置一致**：`tol=1e-10`（=rtol）、`n=40`（=maxterms）、`pencil_rows=128`、`rightanchors=true`、`stepsize=1`。
- **【2026-11-17 追加裁决：`rightanchors` 输入整体删除】**——算法恒按双端锚定（=原 `rightanchors=true`）执行：`|q|>1` 极点一律经恒等式 `q^(d-1) = q^(L-1)·(1/q)^(L-d)` 精确折叠为 `:right` 锚；投影分支（`q/|q|` 压单位圆）**不实现**。理由：投影对超临界极点是有损静默变换（窗内拟合达标、窗外语义错误），GTEMPO 实轴调用点已先行删除 `rightanchors=false`（influence.jl real_factors，守卫 "growing real pole unsupported" 随之激活，见 §6 追记）；GTEMPO 薄壳接入时其自身 `fit_exponentials` 的 `rightanchors` kwarg 一并删除（现调用面已无 false 调用方）。

## 1. 探查定谳（计划赖以成立的关键事实）

**F1【ExpExp 共享循环与 GTEMPO 秩扫同语义】** `_exponential_expansion_impl`（expansion.jl:106-125）：`n = 1..min(alg.n, L÷2)` 逐阶调 `exponential_expansion_n(f, n, alg)`，相对 L2 误差 `≤ alg.tol·‖f‖` 即早退返回最小阶——与 GTEMPO 的 rtol 秩扫描（influence.jl:186-195）语义逐条对应（tol·‖f‖ ⟺ rtol 相对误差）。**秩扫描不需要移植——包装层自带**。

**F2【算法接入点一行委托】** `MatrixPencil` 的全部算法面就是 `exponential_expansion_n(f, p, alg) = matrix_pencil(f, p)`（matrixpencil.jl:77）；独立核心函数与算法结构双面并存是 ExpExp 自身惯例（`matrix_pencil` standalone + `MatrixPencil` struct 同文件）。

**F3【锚定信息过 ExpExp 约定 = 可逆展平】** ExpExp 约定 `f(k) ≈ Σ xs·λ^k`（k=1..L）是单基族；右锚项 `q'^(L−d)` 展平为 `λ = 1/q'`（|λ|>1）乘常数，值恒等；双射可逆（|λ|>1 ⇔ :right 锚、node=1/λ）——**ExpExp 返回约定无需扩展**，锚定由 GTEMPO 消费端还原；同时 ExpExp 侧提供锚定三元组独立函数（见 §2），GTEMPO 走它则无展平/还原往返。

**F4【性能差异点】** 共享循环**每阶重做** H0 的 SVD；GTEMPO 现状是一次 SVD + 逐秩廉价投影。naive 接入 = O(maxterms) 倍铅笔 SVD 成本（40 阶 × 128×L/2 SVD）。两案：n1 共享循环（纯、慢）；n2 算法自带 `exponential_expansion(f, alg::AnchoredMatrixPencil)` 方法（一次 SVD 秩扫，性能/语义与现状平齐）。

**F5【位精确性悬点 = SVD 实现】** GTEMPO 用 `MAK.svd_compact`，ExpExp 现用 `LinearAlgebra`（svd/eigvals）。两者若同 LAPACK 驱动则逐位一致；否则拟合因子位级漂移 → ExactTTI golden 重锚。**前置验证步骤决定工作量**（§3 步骤 0）。

**F6【留在 GTEMPO 的部分】** 退化窗回退（L ≤ maxterms 的单位根精确插值，influence.jl:197-201）是 IF 域策略（"合法有限窗展开"），留 GTEMPO wrapper；`iszero`/`L==1` 边界 ExpExp 已有 iszero、其余随 sweep。

**F7【依赖面】** ExpExp 现依赖仅 LinearAlgebra 系；GTEMPO 未依赖 ExpExp。方案不引入 MatrixAlgebraKit 到 ExpExp（F5 若判同 LAPACK 则 LinearAlgebra 即可）。

## 2. 设计

### 2.1 ExpExp 侧：`src/anchoredpencil.jl`（新文件，include 挂 ExpExp.jl）

```julia
# 1) 独立核心（锚定三元组一等公民；GTEMPO 位精确消费走这层）
"""
    anchored_matrix_pencil(s, p; pencil_rows=128)
        -> (xs, lambdas, anchors)

Hua–Sarkar 铅笔的行子采样 + 双端锚定变体：Hankel 行偏移均匀子采样至
`pencil_rows`（列空间 = 极点 Vandermonde 张成，行选择不变）；拟合极点 |q|>1
折叠为右锚 `1/q`（恒等变换 q^(d−1) = q^(L−1)·(1/q)^(L−d)，权重吸收常数），
返回 (xs, lambdas, anchors)——anchors[i] ∈ {:left, :right}，lambdas 全部 |·|≤1。
"""
function anchored_matrix_pencil(s::Vector{<:Number}, p::Int; pencil_rows::Int=128)

# 2) 秩扫（GTEMPO 现状语义：一次 SVD + 逐秩投影 + 相对误差早退）
"""
    anchored_matrix_pencil_sweep(s; tol=1e-10, n=40, pencil_rows=128)
        -> (xs, lambdas, anchors, err)

一次 H0 SVD 复用的最小阶扫描：首阶相对重建误差 ≤ tol 即返回；扫完不达抛
ArgumentError（带最优 err）。等价于对 p=1..n 逐调 anchored_matrix_pencil，
但只做一次 SVD。
"""
function anchored_matrix_pencil_sweep(s::Vector{<:Number}; tol::Real=1.0e-10, n::Int=40,
                                      pencil_rows::Int=128)

# 3) 算法结构（约定面；默认参数 = GTEMPO 现设置）
struct AnchoredMatrixPencil <: ExponentialExpansionAlgorithm
    n::Int                # 40   = GTEMPO maxterms
    tol::Float64          # 1e-10 = GTEMPO rtol
    verbosity::Int        # 1（共享循环硬要求，§6 批注）
    pencil_rows::Int      # 128
    stepsize::Union{Int,Nothing}   # 1
end
AnchoredMatrixPencil(; n::Int=40, tol::Real=1.0e-10, verbosity::Int=1, pencil_rows::Int=128,
                     stepsize::Union{Int,Nothing}=1) = ...
exponential_expansion_n(f::Vector, p::Int, alg::AnchoredMatrixPencil) =
    展平(anchored_matrix_pencil(f, p; pencil_rows=alg.pencil_rows))
    # 右锚项展平：λ=1/q、xs 吸收常数（F3 双射；值恒等）
```

- export `AnchoredMatrixPencil, anchored_matrix_pencil, anchored_matrix_pencil_sweep`。
- 性能裁定：**GTEMPO 走 `anchored_matrix_pencil_sweep`（F4-n2 语义）**；`AnchoredMatrixPencil` 结构面供一般 ExpExp 用户走共享循环（纯、每阶重做 SVD），docstring 注明该代价。
- 双端锚定恒开、无投影分支（2026-11-17 追加裁决，§0/§6）：不提供 `rightanchors` 输入。

### 2.2 GTEMPO 侧：`fit_exponentials` 变薄壳

```julia
function fit_exponentials(y; rtol=1e-10, maxterms=40, pencil_rows=128)   # 2026-11-17: rightanchors kwarg 删除
    边界（isfinite、L==0/1）原样 →
    (L == 0 || all(iszero, y)) && return (…零因子…)   # 与 ExpExp iszero 语义对齐，保留本地早退
    if L <= maxterms   # 退化窗回退：IF 域策略，留 GTEMPO（F6）
        …单位根插值原样…
    end
    (xs, lambdas, anchors, err) = ExpExp.anchored_matrix_pencil_sweep(y; tol=rtol, n=maxterms,
                                                                      pencil_rows)
    return (; weights=xs, nodes=lambdas, anchors, relative_error=err)
end
```

- 公共签名不变；Project.toml 加 `ExpExp` 依赖；`import ExpExp`（或 `using ExpExp: anchored_matrix_pencil_sweep`）。
- `MAK.svd_compact`/`MAK.eig_vals` 若无其它使用者可从该函数移除（grep 后定）。

### 2.3 位精确性前置闸门（决定验收形态）

- **步骤 0（先做）**：同一组测试矩阵上 `MAK.svd_compact` vs `LinearAlgebra.svd(…; full=false)` 对拍——奇异值/奇异向量逐位比较。
  - 逐位一致 → 移植保持运算序（H0/H1 构造、投影、特征值、LS 的顺序逐行对照），**golden 原样**，验收走 §3-快线；
  - 不一致 → 实现照写，**ExactTTI golden 批次重锚**（六基准 + 双栈对拍新基线），验收走 §3-慢线，工作量 +1 天。

## 3. 验收

1. **ExpExp 单元测试**（新 test 文件）：
   - 解析指数和精确重建（衰减项、双侧窗、复相位极点）；
   - 行子采样 vs 满行：极点一致性（干净数据）；
   - 折叠恒等 round-trip：锚定三元组 vs 展平 (xs, λ) 在窗口上函数值逐位/容差一致；
   - sweep 最小阶语义（rtol 合同）、默认参数值断言（40/1e-10/128）；~~`rightanchors=false` 投影行为~~（2026-11-17 随输入删除移除）；
   - ExpExp 原有 53 项全绿。
2. **GTEMPO**：全测试绿；ExactTTI ≡ PartialIF 对拍与 ED 锚维持（位精确线：逐位不变；漂移线：新基线）。
3. 快线/慢线按 §2.3 步骤 0 结果走。

## 4. 不做清单

- 不改 ExpExp 返回约定（锚定不进共享接口，双射可逆 F3）；
- 不把退化窗回退上收 ExpExp（IF 域策略）；
- 不动 `stepsize=nothing` 级联路径（GTEMPO 恒 stepsize=1）；
- **不实现投影分支、不提供 `rightanchors` 输入**（2026-11-17 追加裁决，§0）；
- BCS/其它调用方零改动。

## 5. 工作量与顺序

- ExpExp：核心移植 + 测试 ≈ 1 天；GTEMPO 薄壳 ≈ 半天；
- 位精确验证（§2.3 步骤 0）≈ 半天；慢线触发时 golden 重锚 +1 天；
- 顺序：步骤 0 → ExpExp → GTEMPO → 全绿。单提交/包。
- 更新 `kernel_and_fit_simplification_plan.md` Part 2 指向本计划（翻案已采纳）。

## 6. 实施结果批注（2026-11-16）

- **ExpExp 侧 ✅ 完成**：`src/anchoredpencil.jl` 三件套（`anchored_matrix_pencil` / `anchored_matrix_pencil_sweep` /
  `AnchoredMatrixPencil`），导出面与测试（test/anchoredpencil.jl 47 项）落地；全套 100 项绿（旧 53 + 新 47）。
  验收按 §3.1：精确重建（衰减/双侧窗/复相位极点）、子采样 vs 满行极点一致、折叠恒等 round-trip
  （展平值恒等 + (权重×λ)/(权重×λ^L) 还原）、sweep 最小阶 + 与逐调逐位等价、投影分支、默认参数断言、参数守卫。
- **struct 面对计划的必要修正**：`AnchoredMatrixPencil` 补 `verbosity` 字段（§2.1 草图缺）——共享循环
  `_exponential_expansion_impl`（expansion.jl:107,114,118）与 `_exponential_expansion_step`（expansion.jl:68）
  读 `alg.verbosity`，缺字段结构面无法走共享循环；默认 1 与姊妹算法一致。
- **步骤 0 定谳【慢线】**：MAK（`svd_compact` 默认 `SafeDivideAndConquer()` = gesdvd）vs `LinearAlgebra.svd`
  （gesdd）：**奇异值逐位一致**（各形状 maxΔσ=0）；**U/Vᴴ 逐位不一致**——同列空间、列相位/符号差 + ~1e-16 噪声
  （逐列 overlap = 1±2e-15，符号对齐后 maxΔU ≈ 6e-17）；`MAK.eig_vals` vs `LinearAlgebra.eigvals` 逐位不一致
  （1×1 除外）。端到端铅笔管线两路 err 同为 1e-15 / 1e-11 量级、收敛秩一致——物理等价、位级漂移。
  → **GTEMPO 接入时 ExactTTI golden 批次重锚**（六基准新基线，物理量仅 ~1e-15 量级移动），验收获收走 §3-慢线。
- GTEMPO 薄壳（§2.2）未动：独立一步，`fit_exponentials` 委托 `anchored_matrix_pencil_sweep`、
  Project.toml 加 ExpExp 依赖、退化窗回退留 wrapper。
- **投影分支裁决（2026-11-17 追记）**：用户裁决 `rightanchors=false` 的静默投影是错误形态——
  GTEMPO 实轴调用点（influence.jl real_factors）**删除 `rightanchors=false` 改用默认 true**：
  超临界极点精确折叠为 `:right`，由既有守卫 "growing real pole unsupported; use PartialIF"
  响抛（该守卫此前被投影架空为死代码）；健康核（极点全 ≤1）折叠惰性、结果逐位不变，
  greater/lesser golden 机器精度锚 10 项实测绿。
- **【2026-11-17 终局裁决：`rightanchors` 输入自 ExpExp 整体删除】**（推翻上一条"保留为通用 API"的
  中间态）——`anchored_matrix_pencil` / `anchored_matrix_pencil_sweep` / `AnchoredMatrixPencil`
  均不再接受 `rightanchors`，行为恒为双端锚定折叠；投影分支自源码删除。用户随后指示实施。
  顺带修复：GTEMPO Project.toml 缺 `[extras]` 段（Random/Test/Printf），本机 registry 快照缺 Printf
  条目故 Pkg.test 沙箱不可用，测试走 `julia --project=. test/runtests.jl` 直跑惯例。
