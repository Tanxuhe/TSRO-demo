# TSRO-DTP: Two-Stage Robust Optimization via Decision Tree Partitioning

## 1. 核心思想：Learning-to-Partition (L2P)

本算法 (**TSRO-DTP**) 的核心理念是：**利用机器学习（决策树）从数据中提取“不确定集的高效划分结构”，从而将复杂的两阶段鲁棒优化问题转化为可求解的大规模 K-Adaptability 问题。**

传统的 $K$-Adaptability 算法试图在优化过程中同时寻找划分（Partition）和策略（Policy），导致计算复杂度随 $K$ 指数级爆炸。TSRO-DTP 将这两个任务解耦：
1.  **机器学习负责“划分”**：利用决策树强大的拟合能力，通过数据驱动的方式快速找到高质量的不确定集划分 $\Omega_1, \dots, \Omega_L$。
2.  **运筹优化负责“策略”**：在固定的划分结构下，利用 MILP 求解器精确寻找每个区域对应的最优策略 $y_l$。

这种解耦使得我们可以使用极高的 $K$ 值（例如 $K=4096$，即深度 12），从而以数量（高分辨率划分）弥补质量（轴对齐划分的几何劣势）。

---

## 2. 算法流程 (Workflow)

目前实现的算法包含两个主要模式：

### 模式 A: 模仿学习 (Standard Mode)
* **目标**：通过模仿一个“老师”（Oracle，如 $K=3$ 的解）来构建划分。
* **步骤**：
    1.  **Oracle 求解**：求解 $K=3$ 的 K-Adaptability 问题，得到第一阶段解 $x_{oracle}$。
    2.  **数据采样**：固定 $x_{oracle}$，采样 $\xi \sim U(\Xi)$，求解 Wait-and-See 问题得到 $y^*(\xi)$。
    3.  **结构提取**：训练 CART 决策树拟合 $\xi \to y^*$，提取叶节点作为划分区域。
    4.  **嵌入求解**：将划分嵌入原问题，求解最终的 $x$ 和 $\{y_l\}$。

### 模式 B: 迭代自进化 (Iterative Mode)
* **目标**：不依赖高质量 Oracle，从简单的静态解 ($K=1$) 开始，通过自我迭代寻找适合当前算法结构的解。
* **步骤**：
    1.  **初始化**：从 $K=1$ 解开始。
    2.  **循环迭代**：
        * 根据当前 $x_t$ 生成数据 $\to$ 训练树 $\to$ 求解得到 $x_{t+1}$。
    3.  **优势**：实验表明，该模式能修正 Oracle 的偏差，使解向“适合轴对齐划分”的方向收敛，最终效果往往优于单纯的模仿。

---

## 3. 当前局限性分析

目前的实验结果显示，在深度极大（Depth=12）时，算法偶尔能超越 $K=3$ 的 Oracle，但大部分情况略逊一筹。主要原因在于**几何失配 (Geometric Mismatch)**：

* **问题本质**：Capital Budgeting 等问题的约束是线性的（$c^T x \le B$），这意味着最优的策略切换边界通常是**超平面（斜面）**。
* **当前瓶颈**：现有的 CART 决策树只能产生**轴对齐（Axis-aligned）**的矩形划分。用“锯齿状”的盒子去逼近光滑的“斜面”效率极低。
    * *比喻*：就像用乐高积木去搭建一个平滑的球体，需要极高的分辨率（极深的树）才能勉强拟合。

---

## 4. 后续工作路线图 (Future Roadmap)

为了将该工作从“验证性实验”提升为“具有理论深度的研究”，后续工作将聚焦于以下两个突破点：

### 🚀 方向一：几何升级——斜决策树 (Oblique Decision Trees)
**目标**：将划分边界从 $x_i \le c$ 升级为 $w^T x \le c$，直接匹配线性约束的几何特性。

* **理论工作**：
    * **Solver 升级**：目前的 `robust_solver.py` 依赖于顶点枚举法（Vertex Enumeration），仅适用于盒式不确定集。对于多面体划分（Polyhedral Partition），需基于**强对偶理论 (Strong Duality)** 重写鲁棒约束。
    * **公式推导**：将 $\max_{\xi \in P} (a^T \xi)$ 转化为对偶形式 $\min b^T \lambda$，从而将半无限约束转化为有限的线性/二阶锥约束。
* **工程工作**：
    * 引入支持斜切分的决策树算法（如 OC1, HHCART 或 Linear Tree）。
    * 重构 `StructureEmbeddedSolver` 类以支持对偶约束。

### 🌐 方向二：泛化升级——跨实例学习 (Cross-Instance Generalization)
**目标**：从 `One-Instance-One-Tree` 进化为 `One-Model-Many-Instances`，训练一个通用的“划分预测器”。

* **核心挑战**：不同实例的参数（$c, r, B$）不同，简单的 $\xi$ 输入无法泛化。
* **解决方案：特征工程 (Feature Engineering)**
    * 设计**上下文特征 (Contextual Features)**，例如：
        * **松弛度 (Slackness)**：当前 $\xi$ 距离预算违约还有多远？
        * **梯度信息 (Gradients)**：$\xi$ 的微小变化对目标函数梯度的影响方向。
        * **相对排名 (Relative Rank)**：项目在当前场景下的性价比排名。
* **训练策略**：
    * 构建“实例池”：生成大量不同参数的 Capital Budgeting 实例。
    * 利用现有的 **Iterative Algorithm** 为每个实例生成高质量标签（Label）。
    * 训练图神经网络 (GNN) 或 Transformer 来预测最优划分规则。

---

## 5. 总结

TSRO-DTP 提出了一种融合机器学习与运筹优化的新范式。目前的实验已验证了“通过学习划分来求解鲁棒优化”的可行性，且迭代模式展现了强大的自适应能力。随着**斜决策树**（解决几何瓶颈）和**跨实例泛化**（解决应用瓶颈）的引入，该算法有望在求解质量和计算效率上同时超越现有的 $K$-Adaptability 基准。
