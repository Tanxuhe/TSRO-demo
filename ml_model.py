from sklearn.tree import DecisionTreeRegressor
import numpy as np

class TreePartitionModel:
    """
    学生模型：利用决策树学习划分结构。
    """
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.leaves = [] 
        if self.max_depth > 0:
            self.clf = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
        else:
            self.clf = None

    def train(self, X, Y):
        if self.max_depth == 0:
            n_features = X.shape[1]
            self.leaves = [{
                'id': 0,
                'bounds': [(-1.0, 1.0) for _ in range(n_features)]
            }]
            return

        self.clf.fit(X, Y)
        self.extract_leaves(X.shape[1]) 

    def extract_leaves(self, n_features):
        self.leaves = []
        tree = self.clf.tree_
        
        def recurse(node_id, current_bounds):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                self.leaves.append({
                    'id': node_id,
                    'bounds': list(current_bounds)
                })
                return
            
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            left_bounds = list(current_bounds)
            lb, ub = left_bounds[feature]
            left_bounds[feature] = (lb, min(ub, threshold))
            recurse(tree.children_left[node_id], left_bounds)
            
            right_bounds = list(current_bounds)
            lb, ub = right_bounds[feature]
            right_bounds[feature] = (max(lb, threshold), ub)
            recurse(tree.children_right[node_id], right_bounds)

        initial_bounds = [(-1.0, 1.0) for _ in range(n_features)]
        recurse(0, initial_bounds)

class RandomPartitionModel:
    """
    对照组模型：完全随机地划分不确定集。
    模拟一个 '瞎猜' 的决策树，用于验证'学习'的有效性。
    """
    def __init__(self, max_depth=3, n_features=4, seed=None):
        self.max_depth = max_depth
        self.n_features = n_features
        self.leaves = []
        self.rng = np.random.RandomState(seed)

    def train(self, X, Y):
        # 忽略 X 和 Y 的数据分布，只利用不确定集的定义域 [-1, 1]
        # 生成一个满二叉树结构的随机划分
        self.leaves = []
        
        # 初始区域：全集 [-1, 1]
        initial_bounds = [(-1.0, 1.0) for _ in range(self.n_features)]
        
        # 队列存储 (当前深度, 当前边界)
        queue = [(0, initial_bounds)]
        
        while queue:
            depth, bounds = queue.pop(0)
            
            if depth < self.max_depth:
                # 随机选择一个切分维度
                dim = self.rng.randint(0, self.n_features)
                lb, ub = bounds[dim]
                
                # 随机选择一个切分点 (Uniform Random Split)
                if abs(ub - lb) > 1e-6:
                    threshold = self.rng.uniform(lb, ub)
                else:
                    threshold = lb
                
                # 左子节点
                left_bounds = list(bounds)
                left_bounds[dim] = (lb, threshold)
                queue.append((depth + 1, left_bounds))
                
                # 右子节点
                right_bounds = list(bounds)
                right_bounds[dim] = (threshold, ub)
                queue.append((depth + 1, right_bounds))
            else:
                # 到达最大深度，作为一个叶子节点
                self.leaves.append({'bounds': bounds})