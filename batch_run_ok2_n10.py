import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gurobipy as gp

# 导入模块
from data_generator import CapitalBudgetingInstance
from oracle import ExactOracleSolver
from ml_model import TreePartitionModel, RandomPartitionModel
from robust_solver import StructureEmbeddedSolver
from benchmarks import KAdaptabilitySolver, ScenarioBasedSolver

# 配置新的缓存目录
CACHE_DIR = "experiment_cache_N10_trials50"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_instance_data(n_items, seed):
    """
    获取实例数据：包含 Benchmarks (Static, K=2, K=3) 和 训练数据 (Teacher K=3)
    """
    filename = os.path.join(CACHE_DIR, f"N{n_items}_seed{seed}.npz")
    
    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        return dict(data)

    # 生成新数据
    ins = CapitalBudgetingInstance(n_projects=n_items, n_factors=4, seed=seed)
    
    cache_data = {
        'N': n_items,
        'Seed': seed
    }

    # 1. 运行 Benchmarks
    # N=10 比较小，求解应该很快
    
    # Static
    t0 = time.time()
    static_solver = KAdaptabilitySolver(ins, K=1)
    res_static = static_solver.solve(time_limit=180)
    cache_data['Obj_Static'] = res_static[0] if res_static else -1.0
    cache_data['T_Static'] = time.time() - t0

    # K=2
    t0 = time.time()
    k2_solver = KAdaptabilitySolver(ins, K=2)
    res_k2 = k2_solver.solve(time_limit=600) 
    cache_data['Obj_K2'] = res_k2[0] if res_k2 else -1.0
    cache_data['T_K2'] = time.time() - t0

    # K=3 (同时作为 Teacher)
    t0 = time.time()
    k3_solver = KAdaptabilitySolver(ins, K=3)
    res_k3 = k3_solver.solve(time_limit=900)
    cache_data['Obj_K3'] = res_k3[0] if res_k3 else -1.0
    cache_data['T_K3'] = time.time() - t0
    
    # 获取 Oracle 解 (x_star from K=3)
    x_star = res_k3[1] if res_k3 else None

    # 2. 生成训练数据 (Teacher = K=3)
    if x_star is not None and cache_data['Obj_K3'] > -1e5:
        oracle = ExactOracleSolver(ins, K=3)
        X_train, Y_train = oracle.generate_training_data(x_star, n_samples=2000)
        cache_data['X_train'] = X_train
        cache_data['Y_train'] = Y_train
        cache_data['Oracle_Valid'] = True
    else:
        cache_data['X_train'] = np.array([])
        cache_data['Y_train'] = np.array([])
        cache_data['Oracle_Valid'] = False

    # 保存缓存
    np.savez(filename, **cache_data)
    return cache_data

def run_experiment(n_items, seed, depth_values):
    # 1. 获取数据
    data = get_instance_data(n_items, seed)
    if data is None or not data['Oracle_Valid']:
        return []

    # 重建实例
    ins = CapitalBudgetingInstance(n_projects=n_items, n_factors=4, seed=seed)
    X_train = data['X_train']
    Y_train = data['Y_train']
    
    # 2. 计算 Upper Bound (Scenario Relaxation)
    # N=10时，1000个场景的覆盖率已经相当不错
    ub_solver = ScenarioBasedSolver(ins, n_scenarios=3000)
    obj_ub = ub_solver.solve()
    
    base_res = {
        'N': n_items,
        'Seed': seed,
        'Obj_Static': float(data['Obj_Static']),
        'Obj_K2': float(data['Obj_K2']),
        'Obj_K3': float(data['Obj_K3']),
        'Obj_UB': obj_ub,
    }
    
    results_list = []
    
    for depth in depth_values:
        res = base_res.copy()
        res['Depth'] = depth
        
        # --- A. Ours (Tree) ---
        t0 = time.time()
        tree_model = TreePartitionModel(max_depth=depth)
        tree_model.train(X_train, Y_train)
        solver_tree = StructureEmbeddedSolver(ins, tree_model)
        obj_tree, _ = solver_tree.solve()
        res['Obj_Tree'] = obj_tree
        res['T_Tree'] = time.time() - t0
        
        # --- B. Random (Random Partition) ---
        # 相同深度，随机划分
        rand_model = RandomPartitionModel(max_depth=depth, n_features=ins.k, seed=seed)
        rand_model.train(None, None)
        solver_rand = StructureEmbeddedSolver(ins, rand_model)
        obj_rand, _ = solver_rand.solve()
        res['Obj_Rand'] = obj_rand
        
        # --- C. 计算 Gap 指标 (Lower Bound vs Upper Bound) ---
        # Gap = (UB - LB) / |LB| * 100%
        # 比较对象：Tree, K=3
        
        if not np.isnan(obj_ub):
            # 你的算法与上界的差距
            res['Gap_Tree_UB'] = (obj_ub - obj_tree) / abs(obj_tree) * 100
            # 随机划分与上界的差距
            res['Gap_Rand_UB'] = (obj_ub - obj_rand) / abs(obj_rand) * 100
            # K=3 (Oracle) 与上界的差距
            res['Gap_K3_UB'] = (obj_ub - res['Obj_K3']) / abs(res['Obj_K3']) * 100
        else:
            res['Gap_Tree_UB'] = np.nan
            res['Gap_K3_UB'] = np.nan

        # 你的算法相对于随机划分的提升 (Value of Learning)
        res['Val_Learn'] = (obj_tree - obj_rand) / abs(obj_rand) * 100
        
        # 你的算法相对于 K=2 的提升 (超越基准)
        res['Impv_vs_K2'] = (obj_tree - res['Obj_K2']) / abs(res['Obj_K2']) * 100

        results_list.append(res)
        
    return results_list

def main():
    # --- 实验设置：N=10, 50个样本 ---
    N_VALUES = [10]
    TRIALS = 50
    # N=10 空间较小，测试深度 2, 3, 4, 5 即可 (4层16个分区, 5层32个分区对于N=10已经很细了)
    DEPTH_VALUES = [2, 3, 4, 5, 6, 7, 8, 9]
    
    all_res = []
    
    print(f"=== Experiment: Small Scale (N={N_VALUES}, Trials={TRIALS}) ===")
    print(f"Goal: Analyze Optimality Gap (vs UB) and Learning Value (vs Random)")
    
    for n in N_VALUES:
        print(f"\n>>> Processing N={n}...")
        for i in tqdm(range(TRIALS)):
            seed = 2024 + i * 100
            try:
                res_list = run_experiment(n, seed, DEPTH_VALUES)
                all_res.extend(res_list)
            except Exception as e:
                print(f"Error seed {seed}: {e}")
                
    df = pd.DataFrame(all_res)
    df.to_csv("batch_results_n10_gap_analysis.csv", index=False)
    
    if not df.empty:
        print("\n" + "="*140)
        print("EXPERIMENT SUMMARY (Averaged)")
        print("Columns: Tree(Ours), Rand(Random), K3(Oracle), UB(Relaxation)")
        print("Gap_UB%: Distance to theoretical limit (Scenario Relaxation)")
        print("Impv_K2%: Improvement over K=2 Benchmark")
        print("="*140)
        
        summary = df.groupby(['N', 'Depth']).agg({
            'Obj_Tree': 'mean',
            'Obj_Rand': 'mean',
            'Obj_K2': 'mean',
            'Obj_K3': 'mean',
            'Obj_UB': 'mean',
            'Gap_Tree_UB': 'mean',
            'Gap_K3_UB': 'mean',
            'Val_Learn': 'mean',
            'Impv_vs_K2': 'mean'
        }).reset_index()
        
        print(f"{'Depth':<6} | {'Leaves':<6} | {'Rand':<8} | {'Tree':<8} | {'K=2':<8} | {'K=3':<8} | {'UB':<8} | {'Gap_Tree%':<10} | {'Gap_K3%':<10} | {'Val_Lrn%':<10}")
        print("-" * 140)
        
        for _, row in summary.iterrows():
            d = int(row['Depth'])
            print(f"{d:<6} | {2**d:<6} | {row['Obj_Rand']:<8.4f} | {row['Obj_Tree']:<8.4f} | {row['Obj_K2']:<8.4f} | {row['Obj_K3']:<8.4f} | {row['Obj_UB']:<8.4f} | {row['Gap_Tree_UB']:<10.2f} | {row['Gap_K3_UB']:<10.2f} | {row['Val_Learn']:<10.2f}")
            
        print("="*140)

if __name__ == "__main__":
    main()