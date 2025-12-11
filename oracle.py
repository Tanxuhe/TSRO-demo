import gurobipy as gp
from gurobipy import GRB
import numpy as np
from benchmarks import KAdaptabilitySolver

class ExactOracleSolver:
    """
    High-Quality Oracle: 
    使用 K=3 的精确 K-Adaptability 算法生成第一阶段解 x*。
    """
    def __init__(self, instance, K=3):
        self.ins = instance
        self.K = K

    def solve(self):
        print(f"    [Oracle] Solving Exact K-Adaptability (K={self.K})... This may take a while.")
        # 给 Oracle 充足的时间 (300s)，确保生成高质量数据
        solver = KAdaptabilitySolver(self.ins, K=self.K)
        obj, x_sol = solver.solve(time_limit=3000)
        
        if x_sol is not None:
            return np.array(x_sol)
        else:
            print("[Oracle] Failed to find optimal x within time limit.")
            return None

    def generate_training_data(self, x_fixed, n_samples=2000):
        """
        数据蒸馏: 固定 x*，生成 (xi, y*) 对。
        """
        if x_fixed is None: return None, None
        
        X_train = []
        Y_train = []
        
        # 批量生成场景
        scenarios = self.ins.rng.uniform(self.ins.xi_lb, self.ins.xi_ub, size=(n_samples, self.ins.k))
        
        for xi in scenarios:
            c_real = self.ins.get_uncertain_cost(xi)
            r_real = self.ins.get_uncertain_profit(xi)
            
            # 检查 x 是否可行 (Budget Check)
            used_budget = np.dot(c_real, x_fixed)
            remaining_budget = self.ins.Budget - used_budget
            
            if remaining_budget < 0:
                # 如果违约，y 只能是 0
                X_train.append(xi)
                Y_train.append(np.zeros(self.ins.n))
                continue

            # 求解确定性 Wait-and-See 问题 (Knapsack)
            m = gp.Model("Sub_Labeling")
            m.setParam('OutputFlag', 0)
            y = m.addVars(self.ins.n, vtype=GRB.BINARY)
            
            # 目标: Maximize second-stage profit
            obj = gp.quicksum(r_real[i] * self.ins.theta * y[i] for i in range(self.ins.n))
            m.setObjective(obj, GRB.MAXIMIZE)
            
            # 预算约束
            m.addConstr(gp.quicksum(c_real[i] * y[i] for i in range(self.ins.n)) <= remaining_budget)
            
            # 互斥约束
            for i in range(self.ins.n):
                if x_fixed[i] > 0.5:
                    m.addConstr(y[i] == 0)
            
            m.optimize()
            
            if m.status == GRB.OPTIMAL:
                y_val = [y[i].X for i in range(self.ins.n)]
                X_train.append(xi)
                Y_train.append(y_val)
            else:
                X_train.append(xi)
                Y_train.append(np.zeros(self.ins.n))
                
        return np.array(X_train), np.array(Y_train)