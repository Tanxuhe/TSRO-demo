import time
import os
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

# 配置缓存目录
CACHE_DIR = "experiment_cache_k2_oracle"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_instance_data(n_items, seed):
    filename = os.path.join(CACHE_DIR, f"N{n_items}_seed{seed}.npz")
    
    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        return dict(data)

    ins = CapitalBudgetingInstance(n_projects=n_items, n_factors=4, seed=seed)
    
    cache_data = {
        'N': n_items,
        'Seed': seed
    }

    # A. Benchmarks
    t0 = time.time()
    static_solver = KAdaptabilitySolver(ins, K=1)
    res_static = static_solver.solve(time_limit=300)
    cache_data['Obj_Static'] = res_static[0] if res_static else -1.0
    cache_data['T_Static'] = time.time() - t0

    t0 = time.time()
    k2_solver = KAdaptabilitySolver(ins, K=2)
    res_k2 = k2_solver.solve(time_limit=1200) 
    cache_data['Obj_K2'] = res_k2[0] if res_k2 else -1.0
    cache_data['T_K2'] = time.time() - t0

    t0 = time.time()
    k3_solver = KAdaptabilitySolver(ins, K=3)
    res_k3 = k3_solver.solve(time_limit=1200)
    cache_data['Obj_K3'] = res_k3[0] if res_k3 else -1.0
    cache_data['T_K3'] = time.time() - t0
    
    # B. Generate Training Data (Oracle K=2)
    x_star = res_k2[1] if res_k2 else None

    if x_star is not None and cache_data['Obj_K2'] > -1e5:
        oracle = ExactOracleSolver(ins, K=2)
        X_train, Y_train = oracle.generate_training_data(x_star, n_samples=2000)
        cache_data['X_train'] = X_train
        cache_data['Y_train'] = Y_train
        cache_data['Oracle_Valid'] = True
    else:
        cache_data['X_train'] = np.array([])
        cache_data['Y_train'] = np.array([])
        cache_data['Oracle_Valid'] = False

    np.savez(filename, **cache_data)
    return cache_data

def run_student_experiment(n_items, seed, depth_values):
    data = get_instance_data(n_items, seed)
    
    if not data['Oracle_Valid']:
        return []

    ins = CapitalBudgetingInstance(n_projects=n_items, n_factors=4, seed=seed)
    X_train = data['X_train']
    Y_train = data['Y_train']
    
    # --- 修复点：添加 T_K3 ---
    base_res = {
        'N': n_items,
        'Seed': seed,
        'Obj_Static': float(data['Obj_Static']),
        'T_Static': float(data['T_Static']),
        'Obj_K2': float(data['Obj_K2']), 
        'T_K2': float(data['T_K2']),
        'Obj_K3': float(data['Obj_K3']),
        'T_K3': float(data['T_K3']),  # 之前漏了这行
    }
    
    results_list = []
    
    for depth in depth_values:
        res = base_res.copy()
        res['Depth'] = depth
        res['Leaves'] = 2**depth
        
        t0 = time.time()
        tree = TreePartitionModel(max_depth=depth)
        tree.train(X_train, Y_train)
        
        student_solver = StructureEmbeddedSolver(ins, tree)
        obj_student, _ = student_solver.solve()
        res['T_Solve'] = time.time() - t0
        res['Obj_Student'] = obj_student
        
        obj_static = res['Obj_Static']
        if abs(obj_static) > 1e-6:
            res['Impv_Student'] = (obj_student - obj_static) / abs(obj_static) * 100
        else:
            res['Impv_Student'] = 0.0
            
        obj_oracle = res['Obj_K2']
        if (obj_oracle - obj_static) > 1e-6:
            res['Gap_Closed_K2'] = (obj_student - obj_static) / (obj_oracle - obj_static) * 100
        else:
            res['Gap_Closed_K2'] = 0.0

        results_list.append(res)
        
    return results_list

def main():
    N_VALUES = [15] 
    TRIALS = 20
    DEPTH_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 
    
    all_res = []
    
    print(f"=== Experiment: Oracle K=2 vs Student (N={N_VALUES}, Trials={TRIALS}) ===")
    
    for n in N_VALUES:
        print(f"\n>>> Processing N={n}...")
        for i in tqdm(range(TRIALS)):
            seed = 2024 + i * 100
            try:
                res_list = run_student_experiment(n, seed, DEPTH_VALUES)
                all_res.extend(res_list)
            except Exception as e:
                print(f"Error seed {seed}: {e}")
                
    df = pd.DataFrame(all_res)
    # 保存结果
    df.to_csv("batch_results_k2_oracle.csv", index=False)
    
    if not df.empty:
        print("\n" + "="*120)
        print("EXPERIMENT SUMMARY (Averaged)")
        print(f"Oracle Base: K=2 (Target Gap=100%). K=3 is shown for reference.")
        print("="*120)
        
        summary = df.groupby(['N', 'Depth']).agg({
            'Obj_Static': 'mean',
            'Obj_K2': 'mean',
            'Obj_K3': 'mean',
            'Obj_Student': 'mean',
            'T_Solve': 'mean',
            'Impv_Student': 'mean',
            'Gap_Closed_K2': 'mean'
        }).reset_index()
        
        print(f"{'Depth':<6} | {'Leaves':<6} | {'Static':<8} | {'Ours':<8} | {'K=2(Orcl)':<10} | {'K=3(Ref)':<8} | {'Time(s)':<8} | {'Impv%':<8} | {'Gap(K2)%':<10}")
        print("-" * 120)
        
        for _, row in summary.iterrows():
            d = int(row['Depth'])
            print(f"{d:<6} | {2**d:<6} | {row['Obj_Static']:<8.4f} | {row['Obj_Student']:<8.4f} | {row['Obj_K2']:<10.4f} | {row['Obj_K3']:<8.4f} | {row['T_Solve']:<8.4f} | {row['Impv_Student']:<8.2f} | {row['Gap_Closed_K2']:<10.2f}")
            
        print("-" * 120)
        # 这里之前报错是因为 df 中没有 T_K3，现在有了
        print(f"Avg Times: Static ~{df['T_Static'].mean():.2f}s, K=2 ~{df['T_K2'].mean():.2f}s, K=3 ~{df['T_K3'].mean():.2f}s")
        print("="*120)

if __name__ == "__main__":
    main()