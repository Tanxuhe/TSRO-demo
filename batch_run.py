import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import gurobipy as gp

# 导入你的模块
from data_generator import CapitalBudgetingInstance
from oracle import ExactOracleSolver
from ml_model import TreePartitionModel
from robust_solver import StructureEmbeddedSolver
from benchmarks import KAdaptabilitySolver

def run_experiment(n_items, seed):
    """运行单个实例的完整对比"""
    # 1. 生成实例
    ins = CapitalBudgetingInstance(n_projects=n_items, n_factors=4, seed=seed)
    
    results = {}
    results['N'] = n_items
    results['Seed'] = seed

    # -------------------------------------------------
    # Baseline 1: Static Robust
    # -------------------------------------------------
    t0 = time.time()
    static_solver = KAdaptabilitySolver(ins, K=1) # K=1 等同于 Static
    obj_static, _ = static_solver.solve()
    results['T_Static'] = time.time() - t0
    results['Obj_Static'] = obj_static
    
    # 如果 Static 都没解出来，这个实例可能太难，跳过
    if obj_static < 0: return None

    # -------------------------------------------------
    # Ours: SD-DTP (Offline Train + Online Solve)
    # -------------------------------------------------
    # A. 离线数据生成 (使用 K=3 的 Oracle)
    t0 = time.time()
    oracle = ExactOracleSolver(ins, K=3)
    x_star = oracle.solve()
    
    if x_star is not None:
        X_train, Y_train = oracle.generate_training_data(x_star, n_samples=2000)
        
        # B. 离线训练
        tree = TreePartitionModel(max_depth=3) # 8 partitions
        tree.train(X_train, Y_train)
        t_offline = time.time() - t0
        results['T_Ours_Offline'] = t_offline

        # C. 在线求解
        t0 = time.time()
        our_solver = StructureEmbeddedSolver(ins, tree)
        obj_ours, _ = our_solver.solve()
        results['T_Ours_Online'] = time.time() - t0
        results['Obj_Ours'] = obj_ours
        
        results['Impv_Ours'] = (obj_ours - obj_static) / obj_static * 100
    else:
        results['Obj_Ours'] = np.nan

    # -------------------------------------------------
    # Exact K-Adaptability (K=2, 3, 4)
    # -------------------------------------------------
    # 设置超时时间，防止 N=30 时卡死
    time_limit = 60 
    
    for k in [2, 3, 4]:
        t0 = time.time()
        k_solver = KAdaptabilitySolver(ins, K=k)
        obj_k, _ = k_solver.solve(time_limit=time_limit)
        
        results[f'T_K{k}'] = time.time() - t0
        results[f'Obj_K{k}'] = obj_k
        
        if obj_k > 0:
            results[f'Impv_K{k}'] = (obj_k - obj_static) / obj_static * 100
        else:
            results[f'Impv_K{k}'] = np.nan # 超时未找到解

    return results

def main():
    # 配置
    N_VALUES = [20, 30]
    TRIALS = 10 # 每个 N 跑 10 个实例
    
    all_res = []
    
    print(f"=== Starting Batch Experiment (Trials={TRIALS}) ===")
    
    for n in N_VALUES:
        print(f"\n>>> Running for N={n}...")
        for i in tqdm(range(TRIALS)):
            seed = 2024 + i * 100
            res = run_experiment(n, seed)
            if res:
                all_res.append(res)
                
    # 转换为 DataFrame
    df = pd.DataFrame(all_res)
    
    # 输出统计信息
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for n in N_VALUES:
        sub_df = df[df['N'] == n]
        print(f"\n[N = {n}]")
        print(f"{'Method':<10} | {'Avg Obj':<10} | {'Avg Time(s)':<12} | {'Avg Impv(%)':<12}")
        print("-" * 55)
        
        print(f"{'Static':<10} | {sub_df['Obj_Static'].mean():<10.4f} | {sub_df['T_Static'].mean():<12.4f} | {'0.0':<12}")
        print(f"{'Ours':<10} | {sub_df['Obj_Ours'].mean():<10.4f} | {sub_df['T_Ours_Online'].mean():<12.4f} | {sub_df['Impv_Ours'].mean():<12.2f}")
        for k in [2, 3, 4]:
            print(f"{f'K={k}':<10} | {sub_df[f'Obj_K{k}'].mean():<10.4f} | {sub_df[f'T_K{k}'].mean():<12.4f} | {sub_df[f'Impv_K{k}'].mean():<12.2f}")

    # 保存详细数据
    df.to_csv("final_experiment_results.csv", index=False)
    print("\nData saved to 'final_experiment_results.csv'")

if __name__ == "__main__":
    main()