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

# 缓存目录 (复用之前的 K=2 缓存)
CACHE_DIR = "experiment_cache_k2_oracle"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_instance_data(n_items, seed):
    """ 获取或生成 Oracle/Benchmark 数据 """
    filename = os.path.join(CACHE_DIR, f"N{n_items}_seed{seed}.npz")
    
    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        return dict(data)

    # 如果缓存不存在，执行完整流程 (这部分逻辑保持不变，用于生成新数据)
    # ... (省略具体生成代码，同上一版，确保能处理新生成的实例) ...
    # 为简洁起见，这里假设缓存已通过之前的运行生成。
    # 如果您需要跑新种子，请把上一版的 get_instance_data 逻辑贴回来。
    return None 

def run_comparative_experiment(n_items, seed, depth_values):
    # 1. 加载数据
    data = get_instance_data(n_items, seed)
    if data is None or not data['Oracle_Valid']:
        return []

    # 重建实例
    ins = CapitalBudgetingInstance(n_projects=n_items, n_factors=4, seed=seed)
    X_train = data['X_train']
    Y_train = data['Y_train']
    
    # 2. 计算 Upper Bound (Scenario Relaxation)
    # 这是一个比较快的计算，可以直接在线跑
    ub_solver = ScenarioBasedSolver(ins, n_scenarios=500)
    obj_ub = ub_solver.solve()
    
    base_res = {
        'N': n_items,
        'Seed': seed,
        'Obj_Static': float(data['Obj_Static']),
        'Obj_K2': float(data['Obj_K2']), # Oracle
        'Obj_UB': obj_ub,
    }
    
    results_list = []
    
    for depth in depth_values:
        res = base_res.copy()
        res['Depth'] = depth
        res['Leaves'] = 2**depth
        
        # --- A. Ours (Decision Tree) ---
        t0 = time.time()
        tree_model = TreePartitionModel(max_depth=depth)
        tree_model.train(X_train, Y_train)
        
        solver_tree = StructureEmbeddedSolver(ins, tree_model)
        obj_tree, _ = solver_tree.solve()
        res['Obj_Tree'] = obj_tree
        res['T_Tree'] = time.time() - t0
        
        # --- B. Random (Random Partition) ---
        # 使用相同的随机种子确保复现，或不同种子多次平均(这里只跑一次)
        t0 = time.time()
        rand_model = RandomPartitionModel(max_depth=depth, n_features=ins.k, seed=seed)
        rand_model.train(None, None) # 随机划分不需要数据
        
        solver_rand = StructureEmbeddedSolver(ins, rand_model)
        obj_rand, _ = solver_rand.solve()
        res['Obj_Rand'] = obj_rand
        res['T_Rand'] = time.time() - t0
        
        # --- C. 计算指标 ---
        
        # 1. 相比 Static 的提升
        base = res['Obj_Static']
        res['Impv_Tree'] = (obj_tree - base) / abs(base) * 100
        res['Impv_Rand'] = (obj_rand - base) / abs(base) * 100
        
        # 2. 决策树相对于随机划分的优势 (Value of Learning)
        # (Obj_Tree - Obj_Rand) / Obj_Rand
        res['Val_Learn'] = (obj_tree - obj_rand) / abs(obj_rand) * 100
        
        # 3. 距离理论最优(UB)的 Gap (越小越好)
        # Gap = (UB - Obj) / Obj
        if not np.isnan(obj_ub):
            res['Gap_Tree_UB'] = (obj_ub - obj_tree) / abs(obj_tree) * 100
            res['Gap_Rand_UB'] = (obj_ub - obj_rand) / abs(obj_rand) * 100
        else:
            res['Gap_Tree_UB'] = np.nan
            res['Gap_Rand_UB'] = np.nan

        results_list.append(res)
        
    return results_list

def main():
    N_VALUES = [15] 
    TRIALS = 20
    DEPTH_VALUES = [2, 3, 4, 5, 6, 7, 8, 9] 
    
    all_res = []
    
    print(f"=== Comparative Exp: Tree vs Random vs UB (N={N_VALUES}) ===")
    
    for n in N_VALUES:
        print(f"\n>>> Processing N={n}...")
        for i in tqdm(range(TRIALS)):
            seed = 2024 + i * 100
            try:
                res_list = run_comparative_experiment(n, seed, DEPTH_VALUES)
                all_res.extend(res_list)
            except Exception as e:
                print(f"Error seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                
    df = pd.DataFrame(all_res)
    df.to_csv("batch_results_comparative.csv", index=False)
    
    if not df.empty:
        print("\n" + "="*140)
        print("EXPERIMENT SUMMARY (Averaged)")
        print("Columns: Tree=Ours, Rand=RandomPartition, UB=ScenarioRelaxation")
        print("Val_Learn%: How much better is Tree than Random?")
        print("Gap_UB%: Distance to theoretical limit (Scenario Relaxation)")
        print("="*140)
        
        summary = df.groupby(['N', 'Depth']).agg({
            'Obj_Static': 'mean',
            'Obj_Tree': 'mean',
            'Obj_Rand': 'mean',
            'Obj_K2': 'mean',
            'Obj_UB': 'mean',
            'Impv_Tree': 'mean',
            'Val_Learn': 'mean',
            'Gap_Tree_UB': 'mean'
        }).reset_index()
        
        # 格式化打印
        print(f"{'Depth':<6} | {'Static':<8} | {'Rand':<8} | {'Tree':<8} | {'K=2':<8} | {'UB':<8} | {'Impv(Tr)%':<10} | {'Val_Lrn%':<10} | {'Gap_UB%':<10}")
        print("-" * 140)
        
        for _, row in summary.iterrows():
            d = int(row['Depth'])
            print(f"{d:<6} | {row['Obj_Static']:<8.4f} | {row['Obj_Rand']:<8.4f} | {row['Obj_Tree']:<8.4f} | {row['Obj_K2']:<8.4f} | {row['Obj_UB']:<8.4f} | {row['Impv_Tree']:<10.2f} | {row['Val_Learn']:<10.2f} | {row['Gap_Tree_UB']:<10.2f}")
            
        print("="*140)

if __name__ == "__main__":
    main()