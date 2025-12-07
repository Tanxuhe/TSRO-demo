import numpy as np

class CapitalBudgetingInstance:
    def __init__(self, n_projects=20, n_factors=4, seed=42):
        self.n = n_projects
        self.k = n_factors
        self.rng = np.random.RandomState(seed)
        
        # 1. 名义参数 (Nominal Parameters)
        # 成本均匀分布 [0, 10]
        self.c_nom = self.rng.uniform(0, 10, self.n) 
        # 收益与成本成正比 (ROI = 20%)
        self.r_nom = self.c_nom / 5.0 
        
        # 2. 因子加载矩阵 (Factor Loading Matrices)
        # 从单位单纯形采样，确保 sum(Phi_i) = 1
        self.Phi = self._generate_simplex_matrix(self.n, self.k) # 影响成本
        self.Psi = self._generate_simplex_matrix(self.n, self.k) # 影响收益
        
        # 3. 资源约束
        # 预算设为总名义成本的 50%
        self.Budget = np.sum(self.c_nom) * 0.5
        
        # 4. 惩罚系数 (Recourse Penalty)
        # 第二阶段收益打8折
        self.theta = 0.8 
        
        # 5. 不确定集 (Box: xi in [-1, 1])
        self.xi_lb = -1.0 * np.ones(self.k)
        self.xi_ub = 1.0 * np.ones(self.k)

    def _generate_simplex_matrix(self, rows, cols):
        mat = self.rng.rand(rows, cols)
        row_sums = mat.sum(axis=1, keepdims=True)
        return mat / row_sums

    def get_uncertain_cost(self, xi):
        # 成本模型: c(xi) = c_nom * (1 + Phi * xi / 2)
        # 最大波动 50%
        factor = 1 + (self.Phi @ xi) * 0.5
        return self.c_nom * factor

    def get_uncertain_profit(self, xi):
        # 收益模型: r(xi) = r_nom * (1 + Psi * xi / 2)
        factor = 1 + (self.Psi @ xi) * 0.5
        return self.r_nom * factor