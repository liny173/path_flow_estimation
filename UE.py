import random
import torch
import numpy as np
import pandas as pd
from collections import defaultdict


# ==========================================
# 1. 传统路径分配求解器 (基于固定路径集的 UE)
# ==========================================
class TraditionalAssignmentSolver:
    def __init__(self, net):
        self.net = net
        self.num_paths = net.num_paths
        self.num_links = net.num_links

        # 提取 OD 对：根据路径的第一个和最后一个 Link ID 定义 OD
        self.od_groups = defaultdict(list)
        for p_idx, p_str in enumerate(net.valid_paths):
            parts = p_str.split('-')
            origin = parts[0]
            destination = parts[-1]
            self.od_groups[(origin, destination)].append(p_idx)

        print(f"Traditional Model: Identified {len(self.od_groups)} OD pairs.")

    def solve_ue(self, total_path_flows, max_iter=200):
        """
        目标：使得同一 OD 对下的所有路径时间趋于相等
        """
        # 1. 计算总的 OD 需求 (根据真实流量汇总，模拟完美的 OD 预测情况)
        od_demand = {}
        for od, p_idxs in self.od_groups.items():
            od_demand[od] = total_path_flows[p_idxs].sum()

        # 2. 初始化：按比例平分流量
        est_path_flows = torch.zeros(self.num_paths)
        for od, p_idxs in self.od_groups.items():
            if len(p_idxs) > 0:
                est_path_flows[p_idxs] = od_demand[od] / len(p_idxs)

        # 3. 迭代优化
        learning_rate = 0.05
        for i in range(max_iter):
            # A. 计算当前路径时间
            link_flows = torch.matmul(self.net.A, est_path_flows)
            link_times = self.net.t0 * (1 + self.net.alpha * torch.pow(link_flows / self.net.capacity, self.net.beta))
            path_times = torch.matmul(self.net.A.t(), link_times)

            new_flows = est_path_flows.clone()

            # B. 对每个 OD 内部进行流量平衡
            for od, p_idxs in self.od_groups.items():
                if len(p_idxs) <= 1:
                    continue

                # 计算该 OD 内的平均时间
                group_times = path_times[p_idxs]
                avg_time = group_times.mean()

                # 时间高于平均的路径减小流量，时间低于平均的增加流量
                diff = (avg_time - group_times) / (avg_time + 1e-6)
                adjustment = learning_rate * diff * od_demand[od]
                new_flows[p_idxs] += adjustment

            # 保持流量非负且总量不变
            est_path_flows = torch.clamp(new_flows, min=0.0)
            # 归一化以确保不违反 OD 总需求
            for od, p_idxs in self.od_groups.items():
                current_sum = est_path_flows[p_idxs].sum()
                if current_sum > 0:
                    est_path_flows[p_idxs] *= (od_demand[od] / current_sum)

            if i % 100 == 0:
                # 评估收敛性
                pass

        return est_path_flows


# ==========================================
# 2. 欠完备需求的传统 UE 求解器
# ==========================================
class IncompleteUESolver:
    def __init__(self, net):
        self.net = net
        self.num_paths = net.num_paths

        # 建立 OD 对映射
        self.od_groups = defaultdict(list)
        for p_idx, p_str in enumerate(net.valid_paths):
            parts = p_str.split('-')
            self.od_groups[(parts[0], parts[-1])].append(p_idx)
        self.all_ods = list(self.od_groups.keys())

    def solve_incomplete_ue(self, true_path_flows, od_coverage=0.6, max_iter=200):
        """
        od_coverage: 0.0 ~ 1.0, 表示模型已知的 OD 比例。覆盖以外的 OD 需求将被视为 0。
        """
        # 1. 计算真实的总 OD 需求
        true_od_demand = {od: true_path_flows[idxs].sum().item() for od, idxs in self.od_groups.items()}

        # 2. 模拟信息缺失：随机采样部分 OD 作为“已知”
        num_known = int(len(self.all_ods) * od_coverage)
        known_ods = random.sample(self.all_ods, num_known)

        # 构造观测到的（残缺）OD 需求
        incomplete_demand = {od: (true_od_demand[od] if od in known_ods else 0.0) for od in self.all_ods}

        # 3. 初始化路径流量 (仅分配已知 OD 的需求)
        est_path_flows = torch.zeros(self.num_paths)
        for od, idxs in self.od_groups.items():
            if incomplete_demand[od] > 0:
                est_path_flows[idxs] = incomplete_demand[od] / len(idxs)

        # 4. 迭代平衡过程 (Frank-Wolfe)
        lr = 0.05
        print(f"UE Solver: Known ODs = {num_known}/{len(self.all_ods)} ({od_coverage * 100:.1f}%)")

        for i in range(max_iter):
            # 计算当前路网耗时
            link_flows = torch.matmul(self.net.A, est_path_flows)
            link_times = self.net.t0 * (1 + self.net.alpha * torch.pow(link_flows / self.net.capacity, self.net.beta))
            path_times = torch.matmul(self.net.A.t(), link_times)

            new_flows = est_path_flows.clone()
            for od, idxs in self.od_groups.items():
                if incomplete_demand[od] <= 0 or len(idxs) <= 1:
                    continue

                # 在已知 OD 内部根据时间进行流量重分配
                group_times = path_times[idxs]
                avg_time = group_times.mean()
                diff = (avg_time - group_times) / (avg_time + 1e-6)
                new_flows[idxs] += lr * diff * incomplete_demand[od]

            # 保持非负与需求一致性
            est_path_flows = torch.clamp(new_flows, min=0.0)
            for od, idxs in self.od_groups.items():
                current_sum = est_path_flows[idxs].sum()
                if incomplete_demand[od] > 0 and current_sum > 0:
                    est_path_flows[idxs] *= (incomplete_demand[od] / current_sum)

        return est_path_flows


# ==========================================
# 3. 运行对比实验
# ==========================================
def run_comparison():

    csv_path = r'D:\PycharmProjects\path_flow\轨迹数据.csv'
    from new_try_4 import RoadNetwork, DataSimulator, solve_fusion  # 不想再copy一下了

    net = RoadNetwork(csv_path)
    simulator = DataSimulator(net, csv_path, pvt_rate=0.05, avi_coverage=0.20)
    data = simulator.generate_snapshot()

    # --- 方法 1: 提出的 Fusion 模型 ---
    print("\n[Method 1] Running MaFusion Solver...")
    estimated_fusion = solve_fusion(net, data)

    od_coverage_rate = 0.65
    # --- 方法 2: 传统 UE 分配模型 ---
    ue_solver = IncompleteUESolver(net)
    estimated_ue = ue_solver.solve_incomplete_ue(
        data['true_path_flows'],
        od_coverage=od_coverage_rate
    )

    # --- 评估 ---
    y_true = data['true_path_flows'].numpy()
    y_fusion = estimated_fusion.numpy()
    y_ue = estimated_ue.numpy()

    def get_metrics(true, pred):
        mae = np.mean(np.abs(true - pred))
        wmape = (np.sum(np.abs(true - pred)) / (np.sum(true) + 1e-6)) * 100
        return mae, wmape

    mae_f, wmape_f = get_metrics(y_true, y_fusion)
    mae_u, wmape_u = get_metrics(y_true, y_ue)

    print("\n" + "=" * 30)
    print(f"{'Metric':<10} | {'Fusion Model':<15} | {'Traditional UE':<15}")
    print("-" * 45)
    print(f"{'MAE':<10} | {mae_f:<15.4f} | {mae_u:<15.4f}")
    print(f"{'WMAPE':<10} | {wmape_f:<14.2f}% | {wmape_u:<14.2f}%")
    print("=" * 30)

    results_df = pd.DataFrame({
        'Path_ID': range(len(y_true)),
        'Path_Composition': net.valid_paths,
        'True_Flow': y_true,
        'Estimated_Flow': y_ue,
        'Error': y_ue - y_true
    })

    # save_path = 'UE_results_场景4.csv'
    # results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    # print("\nResults saved to 'UE_results_场景4.csv'")


if __name__ == "__main__":
    run_comparison()
