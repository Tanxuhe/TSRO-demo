import gurobipy as gp
from gurobipy import GRB
import numpy as np
import itertools

class KAdaptabilitySolver:
    """
    Exact K-Adaptability Solver for Capital Budgeting with Constraint Uncertainty.
    Based on Hanasusanto et al. (2015), Section 3.1, Theorem 5.
    
    Implementation:
    - Uses Gurobi's native bilinear handling (NonConvex=2).
    - Removes artificial bounds on dual variables to ensure feasibility.
    - Focuses on finding feasible solutions quickly (MIPFocus=1).
    """
    def __init__(self, instance, K=2):
        self.ins = instance
        self.K = K

    def solve(self, time_limit=3000):
        # 1. 初始化模型
        m = gp.Model("K_Adapt_Exact_Native_Robust")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', time_limit)
        m.setParam('MIPGap', 1e-4)
        
        # 关键：允许非凸二次约束 (Binary * Continuous)
        m.setParam('NonConvex', 2)
        # 关键：侧重于寻找可行解
        m.setParam('MIPFocus', 1)
        # 避免对偶规约导致的误判
        m.setParam('DualReductions', 0)
        
        N = self.ins.n
        K = self.K
        n_factors = self.ins.k
        epsilon = 1e-3 # 适当放宽 epsilon 以避免数值极其接近导致的不可行
        
        # 2. 主变量
        # x: 第一阶段决策
        x = m.addVars(N, vtype=GRB.BINARY, name="x")
        # y: K个第二阶段策略
        ys = [m.addVars(N, vtype=GRB.BINARY, name=f"y_{k}") for k in range(K)]
        
        # tau: 最坏情况下的 Net Cost (Minimize Max Net Cost <=> Maximize Min Profit)
        # Net Cost = -Profit
        # 给 tau 一个宽松的界
        tau = m.addVar(vtype=GRB.CONTINUOUS, lb=-1e6, ub=1e6, name="tau")
        
        # 3. 互斥约束
        for k in range(K):
            for i in range(N):
                m.addConstr(x[i] + ys[k][i] <= 1)

        # 4. 遍历所有划分 l in {0, 1}^K
        # l_k=0: 策略 k 满足预算 (Feasible)
        # l_k=1: 策略 k 违反预算 (Infeasible)
        partitions = list(itertools.product([0, 1], repeat=K))
        
        for l_idx, l_vec in enumerate(partitions):
            S0 = [k for k, val in enumerate(l_vec) if val == 0] # 可行集
            S1 = [k for k, val in enumerate(l_vec) if val == 1] # 违约集
            
            # --- 对偶变量 (无人工上界) ---
            alpha_pos = m.addVars(n_factors, lb=0, vtype=GRB.CONTINUOUS, name=f"a_pos_{l_idx}")
            alpha_neg = m.addVars(n_factors, lb=0, vtype=GRB.CONTINUOUS, name=f"a_neg_{l_idx}")
            
            beta = m.addVars(S0, lb=0, vtype=GRB.CONTINUOUS, name=f"beta_{l_idx}")
            gamma = m.addVars(S1, lb=0, vtype=GRB.CONTINUOUS, name=f"gamma_{l_idx}")
            
            lam = {}
            if len(S0) > 0:
                lam = m.addVars(S0, lb=0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"lambda_{l_idx}")
                m.addConstr(gp.quicksum(lam[k] for k in S0) == 1)

            # --- 对偶约束 1: 匹配不确定性系数 (Match Coeffs of xi) ---
            # 目标: A^T alpha + sum(beta * C_coeff) + sum(gamma * -C_coeff) = sum(lambda * O_coeff)
            # C_coeff: 成本系数 (正)
            # O_coeff: 净成本系数 (负收益系数)
            
            for j in range(n_factors):
                # Box term
                lhs = alpha_pos[j] - alpha_neg[j]
                
                # Feasible Budget Part: + beta * C_coeff
                for k in S0:
                    term_beta = 0
                    for i in range(N):
                        # C_coeff = c_nom * 0.5 * Phi
                        coef = self.ins.c_nom[i] * 0.5 * self.ins.Phi[i, j]
                        # 累加 (x + y)
                        term_beta += coef * (x[i] + ys[k][i])
                    # 双线性项: beta * (linear expr)
                    lhs += beta[k] * term_beta
                
                # Infeasible Budget Part: + gamma * (-C_coeff)
                # 约束: -Cost <= -Budget - eps. 左边 xi 系数是 -C_coeff.
                for k in S1:
                    term_gamma = 0
                    for i in range(N):
                        coef = self.ins.c_nom[i] * 0.5 * self.ins.Phi[i, j]
                        term_gamma += coef * (x[i] + ys[k][i])
                    # 减去 gamma * term_gamma
                    lhs -= gamma[k] * term_gamma
                
                # RHS: Lambda * O_coeff (Net Cost coeff)
                rhs = 0
                if len(S0) > 0:
                    for k in S0:
                        term_lambda = 0
                        for i in range(N):
                            # Net Cost = -Profit. Coeff = -r_nom * 0.5 * Psi
                            coef = -self.ins.r_nom[i] * 0.5 * self.ins.Psi[i, j]
                            term_lambda += coef * (x[i] + self.ins.theta * ys[k][i])
                        rhs += lam[k] * term_lambda
                
                m.addQConstr(lhs == rhs, name=f"Dual_Match_{l_idx}_{j}")

            # --- 对偶约束 2: 目标值界限 (Objective Bound) ---
            # Dual Objective <= tau (如果 S0 非空)
            # Dual Objective <= -1  (如果 S0 为空，即强制该区域为空)
            
            # Dual Obj = Sum(alpha) + Sum(beta * RHS_Feas) + Sum(gamma * RHS_Infeas)
            # RHS_Feas = Budget - Constant_C
            # RHS_Infeas = Constant_C - Budget - eps
            
            # 1. Box Constant
            term_box = gp.quicksum(alpha_pos[j] + alpha_neg[j] for j in range(n_factors))
            
            # 2. Beta Constant: beta * (Budget - C_const)
            # C_const = sum(c_nom * (x+y)) - Budget
            # So Budget - C_const = 2*Budget - sum(...) 
            # WAIT. Primal: C_coeff xi <= Budget - C_const_raw.
            # Let's stick to: C(xi) <= 0.
            # C(xi) = C_const + C_coeff xi.
            # C_const = sum(...) - Budget.
            # Inequality: C_coeff xi <= -C_const = Budget - sum(...).
            # Dual term: beta * (Budget - sum(...)).
            
            lhs_nonlinear = 0
            for k in S0:
                cost_k = gp.quicksum(self.ins.c_nom[i] * (x[i] + ys[k][i]) for i in range(N))
                # term: beta * (Budget - Cost)
                const_val = self.ins.Budget - cost_k
                lhs_nonlinear += beta[k] * const_val
                
            # 3. Gamma Constant: gamma * (C_const + eps)? No.
            # Primal: -C(xi) <= -eps.
            # -C_const - C_coeff xi <= -eps.
            # -C_coeff xi <= C_const - eps.
            # Dual term: gamma * (C_const - eps).
            # C_const = Cost - Budget.
            # Term: gamma * (Cost - Budget - eps).
            for k in S1:
                cost_k = gp.quicksum(self.ins.c_nom[i] * (x[i] + ys[k][i]) for i in range(N))
                const_val = cost_k - self.ins.Budget - epsilon
                lhs_nonlinear += gamma[k] * const_val
            
            lhs_total = term_box + lhs_nonlinear
            
            if len(S0) > 0:
                # RHS: tau - sum(lambda * O_const)
                # O_const (Net Cost) = sum(-r * (x+theta*y))
                # Inequality: Dual_Obj <= tau - sum(...)
                # => Dual_Obj + sum(...) <= tau
                
                term_lambda_const = 0
                for k in S0:
                    net_cost_const = gp.quicksum(-self.ins.r_nom[i] * (x[i] + self.ins.theta * ys[k][i]) for i in range(N))
                    term_lambda_const += lam[k] * net_cost_const
                
                m.addQConstr(lhs_total + term_lambda_const <= tau, name=f"Dual_Obj_{l_idx}")
            else:
                # S0 is empty -> All strategies violate budget.
                # We must ensure such a region is empty.
                # Farkas condition: Dual_Obj <= -1 (or any negative number).
                m.addQConstr(lhs_total <= -1.0, name=f"Dual_Farkas_{l_idx}")

        # 5. 目标: Maximize Profit => Minimize Net Cost (tau)
        m.setObjective(tau, GRB.MINIMIZE)
        
        m.optimize()
        
        if m.status == GRB.OPTIMAL or (m.status == GRB.TIME_LIMIT and m.SolCount > 0):
            # tau 是 Net Cost (负利润)，所以返回 -tau 作为利润
            return -m.ObjVal, [x[i].X for i in range(N)]
        else:
            print(f"[Oracle] Failed. Status code: {m.status}")
            return -1.0, None
        
class ScenarioBasedSolver:
    """
    Upper Bound Solver: Scenario-based Relaxation.
    
    通过在不确定集中采样有限个场景，求解一个松弛问题。
    其解必然 >= 真实鲁棒最优解，因此构成 Upper Bound。
    """
    def __init__(self, instance, n_scenarios=500):
        self.ins = instance
        self.n_scenarios = n_scenarios
        
    def solve(self, time_limit=60):
        # 1. 采样场景 (包含顶点以增强约束力)
        # 随机采样
        scenarios = self.ins.rng.uniform(
            self.ins.xi_lb, self.ins.xi_ub, 
            size=(self.n_scenarios, self.ins.k)
        )
        # 也可以强制加入全0点或顶点，但在大样本下随机采样足够作为松弛上界
        
        m = gp.Model("Scenario_Relaxation_UB")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', time_limit)
        
        N = self.ins.n
        M = len(scenarios)
        
        # 2. 变量
        # 第一阶段决策 x (全局唯一)
        x = m.addVars(N, vtype=GRB.BINARY, name="x")
        
        # 第二阶段决策 y_j (每个场景 j 独立，Perfect Information)
        ys = [m.addVars(N, vtype=GRB.BINARY, name=f"y_{j}") for j in range(M)]
        
        # 目标值 z (Max Min Profit)
        z = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="z")
        
        # 3. 约束
        for j, xi in enumerate(scenarios):
            # (A) 互斥
            for i in range(N):
                m.addConstr(x[i] + ys[j][i] <= 1)
                
            # (B) 预算约束 (针对该场景必须满足)
            # Cost = c_nom * (1 + 0.5 * Phi * xi) * (x+y)
            # 注意：在 Objective Only 设定下，Phi=0，所以 Cost = c_nom * (x+y)
            c_real = self.ins.get_uncertain_cost(xi)
            cost_expr = gp.quicksum(c_real[i] * (x[i] + ys[j][i]) for i in range(N))
            m.addConstr(cost_expr <= self.ins.Budget)
            
            # (C) 目标约束 (z <= 每个场景的收益)
            r_real = self.ins.get_uncertain_profit(xi)
            profit_expr = gp.quicksum(r_real[i] * (x[i] + self.ins.theta * ys[j][i]) for i in range(N))
            m.addConstr(z <= profit_expr)
            
        # 4. 目标
        m.setObjective(z, GRB.MAXIMIZE)
        
        m.optimize()
        
        if m.status == GRB.OPTIMAL or (m.status == GRB.TIME_LIMIT and m.SolCount > 0):
            return m.ObjVal
        else:
            return np.nan