import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
    def __init__(self, network, csv_path, pvt_rate=0.15, avi_coverage=0.40):
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


# ==========================================
# 3. 融合求解器
# ==========================================
class FusionSolver(nn.Module):
    def __init__(self, num_paths, init_flows, input_feat_dim=3, hidden_dim=64):
        super().__init__()
        self.est_path_flows = nn.Parameter(init_flows.clone())
        self.softplus = nn.Softplus()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(input_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self):
        return self.softplus(self.est_path_flows)


def solve_fusion(net, data):
    A = net.A
    y_avi = data['avi_link_flows']
    mask_avi = data['avi_mask']
    y_pvt = data['pvt_path_flows']
    est_rate = data['global_penetration']

    # 1. 初始化模型 (利用 PVT 流量进行初查)
    initial_guess = y_pvt / (est_rate + 1e-6)
    model = FusionSolver(net.num_paths, initial_guess)
    optimizer = optim.Adam(model.parameters(), lr=0.5)

    # 2. 权重配置
    lambda_avi = 1.0  # 1.0
    lambda_pvt = 5.0  # 5.0
    lambda_bpr = 20.0  # 行为约束权重 20.0
    theta = 0.1  # Logit 灵敏度参数 (值越大对时间越敏感)

    print(f"\nOptimization starting...")

    for step in range(1501):
        optimizer.zero_grad()
        x_pred = model()

        # --- A. 观测 Loss ---
        derived_link_flows = torch.matmul(A, x_pred)
        loss_avi = torch.sum(((derived_link_flows - y_avi) * mask_avi) ** 2) / (mask_avi.sum() + 1e-6)
        loss_pvt = torch.mean((x_pred - initial_guess) ** 2)

        # --- B. 基于 BPR 的路径选择 Loss ---
        # 计算路段行驶时间
        link_times = net.t0 * (1 + net.alpha * torch.pow(derived_link_flows / (net.capacity + 1), net.beta))
        # 计算路径行驶时间
        path_times = torch.matmul(A.t(), link_times)

        loss_ue = torch.tensor(0.0)
        for g_idxs in net.path_choice_groups:
            group_times = path_times[g_idxs]
            group_flows = x_pred[g_idxs]

            # 目标：计算 Logit 分布目标
            logit_target = torch.softmax(-theta * group_times, dim=0)
            # 预测比例
            current_flow_ratio = group_flows / (group_flows.sum() + 1e-6)

            loss_ue += torch.sum((current_flow_ratio - logit_target.detach()) ** 2)

        # 总损失
        loss = lambda_avi * loss_avi + lambda_pvt * loss_pvt + lambda_bpr * loss_ue

        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step:4d} | Total Loss: {loss.item():.2f} | UE Loss: {loss_ue.item():.4f}")

    return model().detach()


# ==========================================
# 4. 主程序运行
# ==========================================
def run_ma_method():
    csv_path = r'D:\PycharmProjects\path_flow\轨迹数据.csv'
    # csv_path = 'pNEUMA_轨迹.csv'

    # 构建网络
    net = RoadNetwork(csv_path)

    # 模拟数据
    simulator = DataSimulator(net, csv_path, pvt_rate=0.15, avi_coverage=0.80)
    data = simulator.generate_snapshot()

    # 模型求解
    estimated_flows = solve_fusion(net, data)

    # 评估与结果保存
    y_true = data['true_path_flows'].numpy()
    y_est = estimated_flows.numpy()

    r2_est = 1 - np.sum((y_true - y_est) ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-8)
    mae = np.mean(np.abs(y_est - y_true))
    wmape = (np.sum(np.abs(y_true - y_est)) / (np.sum(y_true) + 1e-6)) * 100

    print(f"\n=== Final Evaluation ===")
    print(f"MAE: {mae:.4f}")
    print(f"WMAPE: {wmape:.2f}%")
    print(f"R² Score: {r2_est:.4f}")

    results_df = pd.DataFrame({
        'Path_ID': range(len(y_true)),
        'Path_Composition': net.valid_paths,
        'True_Flow': y_true,
        'Estimated_Flow': y_est,
        'Error': y_est - y_true
    })

    # save_path = 'fusion_results_场景4.csv'
    # results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    # print("\nResults saved to 'fusion_results_场景4.csv'")


if __name__ == "__main__":
    run_ma_method()
