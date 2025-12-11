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
CACHE_DIR = "experiment_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_instance_data(n_items, seed):
    """
    获取实例数据。
    如果缓存存在，直接加载；否则运行 Oracle 和 Benchmarks 并保存。
    返回: 字典，包含训练数据和基准测试结果
    """
    filename = os.path.join(CACHE_DIR, f"N{n_items}_seed{seed}.npz")
    
    # 1. 尝试从缓存加载
    if os.path.exists(filename):
        # print(f"Loading cached data for N={n_items}, Seed={seed}...")
        data = np.load(filename, allow_pickle=True)
        return dict(data)

    # 2. 缓存不存在，执行昂贵的计算
    print(f"\n[Generating Data] N={n_items}, Seed={seed} (This may take time...)")
    ins = CapitalBudgetingInstance(n_projects=n_items, n_factors=4, seed=seed)
    
    cache_data = {
        'N': n_items,
        'Seed': seed
    }

    # --- A. 跑 Benchmarks (Static, K=2, K=3) ---
    # 设置超时：K=3 非常慢，给足时间 (例如 20分钟)
    # Static (K=1)
    t0 = time.time()
    static_solver = KAdaptabilitySolver(ins, K=1)
    res_static = static_solver.solve(time_limit=300)
    cache_data['Obj_Static'] = res_static[0] if res_static else -1.0
    cache_data['T_Static'] = time.time() - t0

    # K=2
    t0 = time.time()
    k2_solver = KAdaptabilitySolver(ins, K=2)
    res_k2 = k2_solver.solve(time_limit=1200) # 20 min
    cache_data['Obj_K2'] = res_k2[0] if res_k2 else -1.0
    cache_data['T_K2'] = time.time() - t0

    # K=3 (既是 Benchmark 也是 Oracle 的基础)
    t0 = time.time()
    k3_solver = KAdaptabilitySolver(ins, K=3)
    res_k3 = k3_solver.solve(time_limit=3600) # 20 min
    cache_data['Obj_K3'] = res_k3[0] if res_k3 else -1.0
    cache_data['T_K3'] = time.time() - t0
    
    # 获取 Oracle 解 (x_star)
    # 注意：如果 K=3 求解成功，res_k3[1] 就是 x_star
    x_star = res_k3[1] if res_k3 else None

    # --- B. 生成训练数据 (Oracle) ---
    if x_star is not None and cache_data['Obj_K3'] > -1e5:
        # 使用 Oracle 类来生成数据
        oracle = ExactOracleSolver(ins, K=3)
        # 注意：oracle.solve 内部会再跑一次求解，为了节省时间，我们直接利用上面求出的 x_star
        # 但我们需要调用 generate_training_data
        X_train, Y_train = oracle.generate_training_data(x_star, n_samples=2000)
        
        cache_data['X_train'] = X_train
        cache_data['Y_train'] = Y_train
        cache_data['Oracle_Valid'] = True
    else:
        cache_data['X_train'] = np.array([])
        cache_data['Y_train'] = np.array([])
        cache_data['Oracle_Valid'] = False
        print(f"  Warning: Oracle (K=3) failed for seed {seed}")

    # 3. 保存到缓存
    np.savez(filename, **cache_data)
    print(f"  Saved to {filename}")
    
    return cache_data

def run_student_experiment(n_items, seed, depth_values):
    """
    加载数据，并在不同树深度下运行 Student 模型
    """
    # 1. 获取数据 (自动处理缓存)
    data = get_instance_data(n_items, seed)
    
    if not data['Oracle_Valid']:
        return []

    # 重建实例对象 (因为 npz 不存对象)
    ins = CapitalBudgetingInstance(n_projects=n_items, n_factors=4, seed=seed)
    X_train = data['X_train']
    Y_train = data['Y_train']
    
    # 基础结果 (Benchmark)
    base_res = {
        'N': n_items,
        'Seed': seed,
        'Obj_Static': float(data['Obj_Static']),
        'T_Static': float(data['T_Static']),
        'Obj_K2': float(data['Obj_K2']),
        'T_K2': float(data['T_K2']),
        'Obj_K3': float(data['Obj_K3']),
        'T_K3': float(data['T_K3']),
    }
    
    results_list = []
    
    # 2. 遍历不同的树深度
    for depth in depth_values:
        res = base_res.copy()
        res['Depth'] = depth
        res['Leaves'] = 2**depth
        
        # A. 训练
        t0 = time.time()
        tree = TreePartitionModel(max_depth=depth)
        tree.train(X_train, Y_train)
        res['T_Train'] = time.time() - t0
        
        # B. 求解
        t0 = time.time()
        student_solver = StructureEmbeddedSolver(ins, tree)
        obj_student, _ = student_solver.solve()
        res['T_Solve'] = time.time() - t0
        res['Obj_Student'] = obj_student
        
        # C. 计算提升 (vs Static)
        obj_static = res['Obj_Static']
        if abs(obj_static) > 1e-6:
            res['Impv_Student'] = (obj_student - obj_static) / abs(obj_static) * 100
        else:
            res['Impv_Student'] = 0.0
            
        # D. 计算差距 (vs Oracle K=3)
        # Gap closed = (Student - Static) / (Oracle - Static)
        obj_oracle = res['Obj_K3']
        if (obj_oracle - obj_static) > 1e-6:
            res['Gap_Closed'] = (obj_student - obj_static) / (obj_oracle - obj_static) * 100
        else:
            res['Gap_Closed'] = 0.0

        results_list.append(res)
        
    return results_list

def main():
    # --- 实验配置 ---
    N_VALUES = [20] 
    TRIALS = 10 
    # 对比这几组深度: 3(8叶), 4(16叶), 5(32叶), 6(64叶)
    DEPTH_VALUES = [3, 4, 5, 6] 
    
    all_res = []
    
    print(f"=== Batch Experiment (N={N_VALUES}, Trials={TRIALS}, Depths={DEPTH_VALUES}) ===")
    print(f"Cache Directory: {CACHE_DIR}")
    
    for n in N_VALUES:
        print(f"\n>>> Processing N={n}...")
        for i in tqdm(range(TRIALS)):
            seed = 2024 + i * 100
            try:
                # 这会返回一个列表 (针对每个深度的结果)
                res_list = run_student_experiment(n, seed, DEPTH_VALUES)
                all_res.extend(res_list)
            except Exception as e:
                print(f"Error in seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                
    # 转换为 DataFrame
    df = pd.DataFrame(all_res)
    
    # 保存原始数据
    filename_raw = "batch_results_depths_raw.csv"
    df.to_csv(filename_raw, index=False)
    
    # --- 输出统计报表 ---
    if not df.empty:
        print("\n" + "="*100)
        print("EXPERIMENT SUMMARY (Averaged over trials)")
        print("="*100)
        
        # 我们可以按 (N, Depth) 分组进行统计
        # 先把 Benchmark 的列取出来 (它们对不同 Depth 是重复的，取均值没问题)
        
        summary = df.groupby(['N', 'Depth']).agg({
            'Obj_Static': 'mean',
            'Obj_K2': 'mean',
            'Obj_K3': 'mean',
            'Obj_Student': 'mean',
            'T_Solve': 'mean',
            'Impv_Student': 'mean',
            'Gap_Closed': 'mean'
        }).reset_index()
        
        print(f"{'Depth':<6} | {'Leaves':<6} | {'Static':<8} | {'Ours':<8} | {'K=2':<8} | {'K=3':<8} | {'Time(s)':<8} | {'Impv%':<8} | {'GapClosed%':<10}")
        print("-" * 100)
        
        for _, row in summary.iterrows():
            d = int(row['Depth'])
            leaves = 2**d
            print(f"{d:<6} | {leaves:<6} | {row['Obj_Static']:<8.4f} | {row['Obj_Student']:<8.4f} | {row['Obj_K2']:<8.4f} | {row['Obj_K3']:<8.4f} | {row['T_Solve']:<8.4f} | {row['Impv_Student']:<8.2f} | {row['Gap_Closed']:<10.2f}")
            
        print("-" * 100)
        print(f"(Benchmark Avg Times: Static ~{df['T_Static'].mean():.2f}s, K=2 ~{df['T_K2'].mean():.2f}s, K=3 ~{df['T_K3'].mean():.2f}s)")
        print("="*100)
        print(f"Data saved to '{filename_raw}'")
    else:
        print("No valid results.")

if __name__ == "__main__":
    main()