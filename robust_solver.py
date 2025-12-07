import gurobipy as gp
from gurobipy import GRB

class StructureEmbeddedSolver:
    """
    Ours (SD-DTP) Solver:
    构建并求解一个嵌入了决策树划分结构的单层 MILP。
    """
    def __init__(self, instance, tree_model):
        self.ins = instance
        self.leaves = tree_model.leaves

    def solve(self):
        # 创建模型
        m = gp.Model("Embedded_RO")
        m.setParam('OutputFlag', 0) # 静默模式
        
        # 1. 定义变量
        # x: 第一阶段决策 (全局共享)
        x = m.addVars(self.ins.n, vtype=GRB.BINARY, name="x")
        
        # y: 第二阶段决策 (每个叶子节点有一套独立的 y)
        # y[l, i] 表示在第 l 个区域内，针对项目 i 的补救决策
        y = {}
        for l_idx in range(len(self.leaves)):
            for i in range(self.ins.n):
                y[l_idx, i] = m.addVar(vtype=GRB.BINARY, name=f"y_{l_idx}_{i}")
        
        # z: 辅助变量 (最坏情况下的目标值)
        z = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="z")
        
        # 目标: 最大化最坏情况收益
        m.setObjective(z, GRB.MAXIMIZE)
        
        # 2. 添加约束 (针对每个划分区域 Leaf)
        for l_idx, leaf in enumerate(self.leaves):
            bounds = leaf['bounds'] # 该区域的边界 [(lb, ub), ...]
            
            # --- (A) 互斥约束 ---
            # 一个项目要么第一阶段投，要么第二阶段投，不能重复
            for i in range(self.ins.n):
                m.addConstr(x[i] + y[l_idx, i] <= 1)
            
            # --- (B) 鲁棒预算约束 (Robust Budget) ---
            # 约束: sum(c_i(xi) * (x_i + y_i)) <= Budget, for all xi in Leaf_Box
            # 转化: sum( max_{xi in Box} c_i(xi) * (x_i + y_i) ) <= Budget
            # 因为 x, y >= 0，且 c_i(xi) 关于 xi 的系数非负 (Phi >= 0),
            # 所以 c_i(xi) 的最大值在 xi 取上界 (Upper Bound) 时取得。
            
            xi_worst_cost = [b[1] for b in bounds] # 取每个维度的 UB
            
            lhs = 0
            for i in range(self.ins.n):
                # 计算项目 i 在该区域最坏点处的成本系数
                # c(xi) = c_nom * (1 + Phi * xi / 2)
                factor = 1 + sum(self.ins.Phi[i, k] * xi_worst_cost[k] for k in range(self.ins.k)) * 0.5
                coeff = self.ins.c_nom[i] * factor
                lhs += coeff * (x[i] + y[l_idx, i])
                
            m.addConstr(lhs <= self.ins.Budget, name=f"Budget_Leaf_{l_idx}")
            
            # --- (C) 鲁棒目标约束 (Robust Profit) ---
            # 约束: z <= sum(r_i(xi) * (x_i + theta * y_i)), for all xi in Leaf_Box
            # 转化: z <= min_{xi in Box} sum(...)
            # 同样利用单调性，收益最小值在 xi 取下界 (Lower Bound) 时取得。
            
            xi_worst_profit = [b[0] for b in bounds] # 取每个维度的 LB
            
            rhs = 0
            for i in range(self.ins.n):
                # r(xi) = r_nom * (1 + Psi * xi / 2)
                factor = 1 + sum(self.ins.Psi[i, k] * xi_worst_profit[k] for k in range(self.ins.k)) * 0.5
                coeff = self.ins.r_nom[i] * factor
                rhs += coeff * (x[i] + self.ins.theta * y[l_idx, i])
                
            m.addConstr(z <= rhs, name=f"Obj_Leaf_{l_idx}")
            
        # 3. 求解
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            # 返回: (目标值, 第一阶段解 x)
            x_sol = [x[i].X for i in range(self.ins.n)]
            return m.ObjVal, x_sol
        else:
            return -1.0, None