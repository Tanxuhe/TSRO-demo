import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# 导入模块
from data_generator import CapitalBudgetingInstance
from oracle import ExactOracleSolver
from ml_model import TreePartitionModel
from robust_solver import StructureEmbeddedSolver
from benchmarks import KAdaptabilitySolver

def validate_solution(instance, x_sol, n_test=10000):
    """
    公平验证器: 
    在 10,000 个随机场景下测试解 x 的表现。
    计算: 1. 可行率 (Feasibility)  2. 真实最坏收益 (Real Worst-Case Profit)
    """
    if x_sol is None:
        return 0.0, -9999.0

    x_vec = np.array(x_sol)
    # 修正数值精度，确保为 0/1 整数
    x_vec = (x_vec > 0.5).astype(int)
    
    violation_count = 0
    min_profit = float('inf')
    
    # 批量生成测试场景
    scenarios = instance.rng.uniform(instance.xi_lb, instance.xi_ub, size=(n_test, instance.k))
    
    # 针对每个场景进行测试
    for xi in scenarios:
        c_real = instance.get_uncertain_cost(xi)
        r_real = instance.get_uncertain_profit(xi)
        
        # 1. 检查第一阶段预算
        cost_x = np.dot(c_real, x_vec)
        
        if cost_x > instance.Budget:
            # 严重违约: 第一阶段就超支了
            violation_count += 1
            current_profit = -1000.0 # 惩罚值
        else:
            # 2. 计算最优补救策略 y (Wait-and-See)
            # 这是一个简单的 Knapsack 问题，我们可以快速求解
            remaining_budget = instance.Budget - cost_x
            
            # 使用 Gurobi 求解子问题 (保证准确性)
            m_val = gp.Model("Val_Sub")
            m_val.setParam('OutputFlag', 0)
            y = m_val.addVars(instance.n, vtype=GRB.BINARY)
            
            # Maximize profit
            obj = gp.quicksum(r_real[i] * instance.theta * y[i] for i in range(instance.n))
            m_val.setObjective(obj, GRB.MAXIMIZE)
            
            # Budget constraint
            m_val.addConstr(gp.quicksum(c_real[i] * y[i] for i in range(instance.n)) <= remaining_budget)
            
            # Mutually exclusive
            for i in range(instance.n):
                if x_vec[i] == 1:
                    m_val.addConstr(y[i] == 0)
            
            m_val.optimize()
            
            if m_val.status == GRB.OPTIMAL:
                y_profit = m_val.ObjVal
                current_profit = np.dot(r_real, x_vec) + y_profit
            else:
                current_profit = -1000.0

        if current_profit < min_profit:
            min_profit = current_profit
            
    feasibility_rate = 100.0 * (1 - violation_count / n_test)
    if feasibility_rate < 100:
        min_profit = -1000.0
        
    return feasibility_rate, min_profit

def main():
    # --- 实验配置 ---
    N_PROJECTS = 20
    N_FACTORS = 4
    TREE_DEPTH = 3  # 3层树 = 8个分区
    
    # Oracle 和 Benchmark 配置
    ORACLE_K = 3      # 老师用 K=3
    BENCHMARK_K = 2   # 对手用 K=2
    
    print(f"=== Experiment: Capital Budgeting (N={N_PROJECTS}) ===")
    print(f"Setup: Oracle(K={ORACLE_K}), Benchmark(K={BENCHMARK_K}), Student(Depth={TREE_DEPTH})")
    
    # 1. 生成题目 (Instance)
    ins = CapitalBudgetingInstance(n_projects=N_PROJECTS, n_factors=N_FACTORS, seed=2024)
    
    # -------------------------------------------------
    # Phase 1: 离线数据生成 (Teacher: Exact K=3)
    # -------------------------------------------------
    print("\n[1] Offline: Oracle Generating Data...")
    t0 = time.time()
    oracle = ExactOracleSolver(ins, K=ORACLE_K)
    x_star = oracle.solve()
    
    # 生成训练集 (Wait-and-See)
    X_train, Y_train = oracle.generate_training_data(x_star, n_samples=2000)
    print(f"    Oracle Time: {time.time()-t0:.2f}s | Training Samples: {len(X_train)}")
    
    if x_star is None:
        print("Oracle failed. Aborting.")
        return

    # -------------------------------------------------
    # Phase 2: 离线结构学习 (Student)
    # -------------------------------------------------
    print("\n[2] Offline: Structure Learning...")
    t0 = time.time()
    tree_model = TreePartitionModel(max_depth=TREE_DEPTH)
    tree_model.train(X_train, Y_train)
    print(f"    Training Time: {time.time()-t0:.4f}s")
    
    # -------------------------------------------------
    # Phase 3: 在线求解 (Ours: SD-DTP)
    # -------------------------------------------------
    print("\n[3] Online: Solving with SD-DTP (Ours)...")
    t0 = time.time()
    our_solver = StructureEmbeddedSolver(ins, tree_model)
    our_obj, our_x = our_solver.solve()
    our_time = time.time() - t0
    print(f"    Ours Time: {our_time:.4f}s | Nominal Obj: {our_obj:.4f}")
    
    # -------------------------------------------------
    # Phase 4: 基准对比 (Benchmarks)
    # -------------------------------------------------
    print("\n[4] Online: Solving Benchmarks...")
    
    # Baseline 1: Static Robust
    t0 = time.time()
    static_model = TreePartitionModel(max_depth=0)
    static_model.train(X_train[:10], Y_train[:10]) # Dummy init
    static_solver = StructureEmbeddedSolver(ins, static_model)
    static_obj, static_x = static_solver.solve()
    static_time = time.time() - t0
    print(f"    Static Time: {static_time:.4f}s | Obj: {static_obj:.4f}")
    
    # Baseline 2: Exact K-Adaptability (K=2)
    t0 = time.time()
    k_solver = KAdaptabilitySolver(ins, K=BENCHMARK_K)
    k_obj, k_x = k_solver.solve(time_limit=60) # 限制在线求解时间
    k_time = time.time() - t0
    print(f"    K-Adapt(K={BENCHMARK_K}) Time: {k_time:.4f}s | Obj: {k_obj:.4f}")

    # -------------------------------------------------
    # Phase 5: 终极验证 (Validation)
    # -------------------------------------------------
    print("\n" + "="*80)
    print("VALIDATION PHASE: Stress testing on 10,000 random scenarios")
    print("="*80)
    
    feas_our, prof_our = validate_solution(ins, our_x)
    feas_static, prof_static = validate_solution(ins, static_x)
    feas_k, prof_k = validate_solution(ins, k_x)
    
    # -------------------------------------------------
    # 汇总报表
    # -------------------------------------------------
    print("\n" + "="*80)
    print(f"{'Method':<15} | {'Time(s)':<8} | {'Nominal Obj':<12} | {'Real Feas.':<10} | {'Real Min Profit':<15}")
    print("-" * 80)
    print(f"{'Ours (Tree)':<15} | {our_time:<8.4f} | {our_obj:<12.4f} | {feas_our:<8.1f}%   | {prof_our:<15.4f}")
    print(f"{f'K-Adapt(K={BENCHMARK_K})':<15} | {k_time:<8.4f} | {k_obj:<12.4f} | {feas_k:<8.1f}%   | {prof_k:<15.4f}")
    print(f"{'Static':<15} | {static_time:<8.4f} | {static_obj:<12.4f} | {feas_static:<8.1f}%   | {prof_static:<15.4f}")
    print("="*80)
    
    if prof_static > 0:
        imp = (prof_our - prof_static) / prof_static * 100
        print(f"\n>>> Final Conclusion: Ours improves {imp:.2f}% over Static (in Real Profit).")

if __name__ == "__main__":
    main()