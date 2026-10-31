import torch
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from collections import defaultdict, Counter


# ==========================================
# 1. 基础拓扑构建
# ==========================================
class RoadNetwork:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=['Vehicle ID', 'Link ID'])

        self.real_ids = sorted(df['Link ID'].unique())
        self.num_links = len(self.real_ids)
        self.link_id_map = {rid: i for i, rid in enumerate(self.real_ids)}

        # --- BPR 参数初始化 ---
        self.t0 = torch.ones(self.num_links) * 1.0
        self.capacity = torch.ones(self.num_links) * 1200.0
        self.alpha = 0.15
        self.beta = 4.0

        # --- 提取路径 ---
        df = df.sort_values(by=['Vehicle ID', 'Timestep'])
        vehicle_seqs = df.groupby('Vehicle ID')['Link ID'].agg(list)

        path_counter = Counter()
        for seq in vehicle_seqs:
            # 去除重复值并保留顺序
            seq_u = sorted(set(seq), key=seq.index)
            if len(seq_u) > 1:
                seq_string = "-".join(str(element) for element in seq_u)
                path_counter[seq_string] += 1

        self.valid_paths = sorted(list(path_counter.keys()))
        self.num_paths = len(self.valid_paths)
        self.path_map = {p: i for i, p in enumerate(self.valid_paths)}

        # --- 关联矩阵 A (Link-Path) ---
        self.A = torch.zeros(self.num_links, self.num_paths)
        for p_idx, p_str in enumerate(self.valid_paths):
            nodes = [float(x) for x in p_str.split('-')]
            for node_id in nodes:
                if node_id in self.link_id_map:
                    self.A[self.link_id_map[node_id], p_idx] = 1.0

        # --- 路径组构建 (用于 UE-BPR 约束) ---
        # 将具有相同“入口路段(起点)”的路径归为一组，模拟交通分配中的路径选择
        origin_groups = defaultdict(list)
        for p_idx, p_str in enumerate(self.valid_paths):
            origin_link = p_str.split('-')[0]
            origin_groups[origin_link].append(p_idx)

        # 存在多条竞争路径的起点才需要加入约束
        self.path_choice_groups = [idxs for idxs in origin_groups.values() if len(idxs) > 1]

        # print(f"Network: {self.num_links} Links, {self.num_paths} Paths extracted.")


# ==========================================
# 2. 数据模拟
# ==========================================
class DataSimulator:
    def __init__(self, network, csv_path, pvt_rate=0.15, avi_coverage=0.30):
        self.net = network
        self.df = pd.read_csv(csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]
        self.df = self.df.dropna(subset=['Vehicle ID', 'Link ID'])
        self.pvt_rate = pvt_rate
        self.avi_coverage = avi_coverage

    def generate_snapshot(self):
        all_vehs = self.df['Vehicle ID'].unique()
        veh_paths = self.df.groupby('Vehicle ID')['Link ID'].agg(list)

        true_path_flows = torch.zeros(self.net.num_paths)
        for vid, seq in veh_paths.items():
            seq_u = sorted(set(seq), key=seq.index)
            if len(seq_u) > 1:
                seq_string = "-".join(str(element) for element in seq_u)
                if seq_string in self.net.path_map:
                    true_path_flows[self.net.path_map[seq_string]] += 1

        # 采样部分车辆作为 PVT 数据
        probe_vehs = np.random.choice(all_vehs, size=int(len(all_vehs) * self.pvt_rate), replace=False)
        pvt_path_flows = torch.zeros(self.net.num_paths)
        for vid in probe_vehs:
            seq = veh_paths[vid]
            seq_u = sorted(set(seq), key=seq.index)
            if len(seq_u) > 1:
                seq_string = "-".join(str(element) for element in seq_u)
                if seq_string in self.net.path_map:
                    pvt_path_flows[self.net.path_map[seq_string]] += 1

        # 路段流量计算与 AVI 掩码
        true_link_flows = torch.matmul(self.net.A, true_path_flows)
        num_sensors = max(1, int(self.net.num_links * self.avi_coverage))
        sensor_indices = np.random.choice(self.net.num_links, size=num_sensors, replace=False)
        avi_mask = torch.zeros(self.net.num_links)
        avi_mask[sensor_indices] = 1.0

        return {
            'true_path_flows': true_path_flows,
            'pvt_path_flows': pvt_path_flows,
            'avi_link_flows': true_link_flows * avi_mask,
            'avi_mask': avi_mask,
            'global_penetration': self.pvt_rate,
        }


def run_ma_method():
    csv_path = r'D:\PycharmProjects\path_flow\轨迹数据.csv'

    # 构建网络
    net = RoadNetwork(csv_path)

    # 模拟数据
    simulator = DataSimulator(net, csv_path, pvt_rate=0.15, avi_coverage=0.80)
    data = simulator.generate_snapshot()

    # =========================================================
    # 1. 构建特征（PVT + AVI）
    # =========================================================

    A = net.A.numpy()
    num_links, num_paths = A.shape

    # AVI观测路段流量
    y_avi = data['avi_link_flows'].numpy()
    mask_avi = data['avi_mask'].numpy()

    # 路段->路径映射特征（只取被观测的路段）
    X_avi = (A.T * y_avi).T
    X_avi = X_avi[mask_avi > 0, :].T  # (num_paths, num_avi_links)

    # PVT路径流量
    y_pvt = data['pvt_path_flows'].numpy().reshape(-1, 1)

    # 拼接特征
    X = np.hstack([X_avi, y_pvt])

    # 目标：真实路径流量
    y = data['true_path_flows'].numpy()

    # =========================================================
    # 2. 构建 LightGBM 数据集
    # =========================================================
    train_data = lgb.Dataset(X, label=y)

    # =========================================================
    # 3. 设置模型参数
    # =========================================================
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    # =========================================================
    # 4. 训练模型
    # =========================================================
    print("Training LightGBM...")
    model = lgb.train(params, train_data, num_boost_round=500)

    # =========================================================
    # 5. 预测
    # =========================================================
    y_pred = model.predict(X)

    # 保证非负（路径流量必须 >= 0）
    y_pred = np.maximum(y_pred, 0)

    # =========================================================
    # 6. 评估
    # =========================================================
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    wmape = np.sum(np.abs(y - y_pred)) / (np.sum(y) + 1e-6) * 100

    print("\n=== LightGBM Path Flow Estimation ===")
    print(f"MAE: {mae:.4f}")
    print(f"WMAPE: {wmape:.2f}%")
    print(f"R² Score: {r2:.4f}")

    # =========================================================
    # 7. 保存结果
    # =========================================================
    results_df = pd.DataFrame({
        'Path_ID': np.arange(len(y)),
        'Path_Composition': net.valid_paths,
        'True_Flow': y,
        'Estimated_Flow': y_pred,
        'Error': y_pred - y
    })

    results_df.to_csv('lgbm_path_flow_results_场景4.csv', index=False)


if __name__ == "__main__":
    run_ma_method()
