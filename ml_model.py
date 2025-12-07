from sklearn.tree import DecisionTreeRegressor
import numpy as np

class TreePartitionModel:
    """
    学生模型：
    利用决策树学习输入场景 xi 到最优策略 y* 的映射。
    核心产出不是预测值，而是树的叶子节点定义的划分结构 (Partitions)。
    """
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.leaves = [] # 存储划分结构: [{'id': int, 'bounds': [(lb, ub), ...]}, ...]
        
        # 只有当深度 > 0 时才初始化 sklearn 模型
        if self.max_depth > 0:
            # 使用多输出回归树 (Multi-output Regression Tree)
            self.clf = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
        else:
            self.clf = None

    def train(self, X, Y):
        # 特殊情况：静态鲁棒 (Static Robust)
        # 如果深度为 0，不训练，直接返回一个包含全域的大盒子
        if self.max_depth == 0:
            n_features = X.shape[1]
            # 假设 xi 的范围是 [-1, 1] (与 data_generator 一致)
            self.leaves = [{
                'id': 0,
                'bounds': [(-1.0, 1.0) for _ in range(n_features)]
            }]
            return

        # 正常训练
        self.clf.fit(X, Y)
        # 提取结构
        self.extract_leaves(X.shape[1]) 

    def extract_leaves(self, n_features):
        """
        解析 sklearn 的决策树结构，将每个叶子节点转化为一个超矩形 (Hyper-rectangle)。
        """
        self.leaves = []
        tree = self.clf.tree_
        
        # 递归遍历树以获取每个节点的边界
        def recurse(node_id, current_bounds):
            # 如果是叶子节点
            if tree.children_left[node_id] == tree.children_right[node_id]:
                self.leaves.append({
                    'id': node_id,
                    'bounds': list(current_bounds) # 深拷贝当前边界
                })
                return
            
            # 如果是分支节点
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            # 左分支: xi <= threshold
            # 更新该特征的上界
            left_bounds = list(current_bounds)
            lb, ub = left_bounds[feature]
            # 取交集: min(原上界, 阈值)
            left_bounds[feature] = (lb, min(ub, threshold))
            recurse(tree.children_left[node_id], left_bounds)
            
            # 右分支: xi > threshold
            # 更新该特征的下界
            right_bounds = list(current_bounds)
            lb, ub = right_bounds[feature]
            # 取交集: max(原下界, 阈值)
            right_bounds[feature] = (max(lb, threshold), ub)
            recurse(tree.children_right[node_id], right_bounds)

        # 初始边界：整个不确定集 [-1, 1]
        initial_bounds = [(-1.0, 1.0) for _ in range(n_features)]
        recurse(0, initial_bounds)
        print(f"    [Tree] Structure extracted. Total partitions (leaves): {len(self.leaves)}")