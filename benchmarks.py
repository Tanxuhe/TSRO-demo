import gurobipy as gp
from gurobipy import GRB
import numpy as np

class KAdaptabilitySolver:
    """
    Exact K-Adaptability Solver based on Hanasusanto et al. (2015).
    Ref: Appendix 6.2 "Robust capital budgeting K-adaptability model"
    
    Variables mapping:
    - x: First-stage decisions
    - y[k]: Second-stage policy k
    - rho[k]: Weight of policy k (dual variable lambda in paper)
    - z[k]: Linearization of rho[k] * y[k]
    """
    def __init__(self, instance, K=2):
        self.ins = instance
        self.K = K

    def solve(self, time_limit=300):
        m = gp.Model("K_Adapt_Exact")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', time_limit)
        # 提高精度，防止数值误差导致的可行解丢失
        m.setParam('MIPGap', 1e-4) 
        
        N = self.ins.n
        K = self.K
        
        # --- Variables ---
        x = m.addVars(N, vtype=GRB.BINARY, name="x")
        y = m.addVars(K, N, vtype=GRB.BINARY, name="y")
        rho = m.addVars(K, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="rho")
        
        # Linearization variable z[k, i] = rho[k] * y[k, i]
        z = m.addVars(K, N, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z")
        
        # Auxiliary variables for absolute values in objective (dual norm)
        # u[j] + v[j] >= |coeff_j|
        uv_sum = m.addVars(self.ins.k, lb=0, vtype=GRB.CONTINUOUS, name="uv_sum")

        # --- Constraints ---
        
        # 1. Simplex constraint for rho
        m.addConstr(gp.quicksum(rho[k] for k in range(K)) == 1.0)
        
        # 2. Linearization constraints (McCormick for binary * continuous)
        # z = rho * y
        for k in range(K):
            for i in range(N):
                m.addConstr(z[k, i] <= rho[k])
                m.addConstr(z[k, i] <= y[k, i])
                m.addConstr(z[k, i] >= rho[k] + y[k, i] - 1)
        
        # 3. Robust Feasibility (Budget Constraint)
        # Each policy y^k must be feasible for the WORST-CASE cost
        # c_i(xi) = c_nom * (1 + Phi * xi / 2). Max when xi = 1 (assuming Phi >= 0)
        worst_cost = np.zeros(N)
        for i in range(N):
            factor = 1.0 + np.sum(self.ins.Phi[i, :]) * 0.5
            worst_cost[i] = self.ins.c_nom[i] * factor
            
        for k in range(K):
            # Mutually exclusive x + y <= 1
            for i in range(N):
                m.addConstr(x[i] + y[k, i] <= 1)
            # Budget
            m.addConstr(gp.quicksum(worst_cost[i] * (x[i] + y[k, i]) for i in range(N)) <= self.ins.Budget)

        # --- Objective Function ---
        # Obj = min_xi max_k Profit(xi, k)
        # Dualized form: sum(rho_k * Const_k) - sum(|Coeff_j|)
        # Profit_k(xi) = sum_i r_i(xi) * (x_i + theta * y_ki)
        # r_i(xi) = r_nom * (1 + Psi * xi / 2)
        
        # Constant part (Nominal Profit weighted by rho)
        # = sum_i r_nom * (x_i * sum(rho) + theta * sum(z_ki))
        # Since sum(rho) = 1, this simplifies to:
        term_constant = gp.LinExpr()
        for i in range(N):
            term_constant += self.ins.r_nom[i] * x[i]
            for k in range(K):
                term_constant += self.ins.r_nom[i] * self.ins.theta * z[k, i]

        # Coefficient part (Sensitivity to xi_j)
        # Coeff_j = sum_i [ r_nom * 0.5 * Psi_ij * (x_i + theta * sum_k z_ki) ]
        for j in range(self.ins.k):
            term_coeff = gp.LinExpr()
            for i in range(N):
                val = self.ins.r_nom[i] * 0.5 * self.ins.Psi[i, j]
                term_coeff += val * x[i]
                for k in range(K):
                    term_coeff += val * self.ins.theta * z[k, i]
            
            # uv_sum[j] >= |term_coeff|
            m.addConstr(uv_sum[j] >= term_coeff)
            m.addConstr(uv_sum[j] >= -term_coeff)

        # Final Objective
        m.setObjective(term_constant - gp.quicksum(uv_sum[j] for j in range(self.ins.k)), GRB.MAXIMIZE)

        m.optimize()
        
        if m.status == GRB.OPTIMAL or (m.status == GRB.TIME_LIMIT and m.SolCount > 0):
            return m.ObjVal, [x[i].X for i in range(N)]
        else:
            return -1.0, None