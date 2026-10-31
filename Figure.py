import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_1 = pd.read_csv('fusion_results_场景1.csv')
df_2 = pd.read_csv('fusion_results_场景2.csv')
df_3 = pd.read_csv('fusion_results_场景3.csv')
df_4 = pd.read_csv('fusion_results_场景4.csv')

dt_1 = pd.read_csv('UE_results_场景1.csv')
dt_2 = pd.read_csv('UE_results_场景2.csv')
dt_3 = pd.read_csv('UE_results_场景3.csv')
dt_4 = pd.read_csv('UE_results_场景4.csv')

d_1 = pd.read_csv('lgbm_path_flow_results_场景1.csv')
d_2 = pd.read_csv('lgbm_path_flow_results_场景2.csv')
d_3 = pd.read_csv('lgbm_path_flow_results_场景3.csv')
d_4 = pd.read_csv('lgbm_path_flow_results_场景4.csv')

error_1 = df_1['Error'].values
abs_error_1 = np.abs(error_1)
error_1_2 = dt_1['Error'].values
abs_error_1_2 = np.abs(error_1_2)
error_1_3 = d_1['Error'].values
abs_error_1_3 = np.abs(error_1_3)

t_1 = df_1['True_Flow'].values
e_1 = df_1['Estimated_Flow'].values
e_1_2 = dt_1['Estimated_Flow'].values
e_1_3 = d_1['Estimated_Flow'].values

error_2 = df_2['Error'].values
abs_error_2 = np.abs(error_2)
error_2_2 = dt_2['Error'].values
abs_error_2_2 = np.abs(error_2_2)
error_2_3 = d_2['Error'].values
abs_error_2_3 = np.abs(error_2_3)

e_2 = df_2['Estimated_Flow'].values
e_2_2 = dt_2['Estimated_Flow'].values
e_2_3 = d_2['Estimated_Flow'].values

error_3 = df_3['Error'].values
abs_error_3 = np.abs(error_3)
error_3_2 = dt_3['Error'].values
abs_error_3_2 = np.abs(error_3_2)
error_3_3 = d_3['Error'].values
abs_error_3_3 = np.abs(error_3_3)

e_3 = df_3['Estimated_Flow'].values
e_3_2 = dt_3['Estimated_Flow'].values
e_3_3 = d_3['Estimated_Flow'].values

error_4 = df_4['Error'].values
abs_error_4 = np.abs(error_4)
error_4_2 = dt_4['Error'].values
abs_error_4_2 = np.abs(error_4_2)
error_4_3 = d_4['Error'].values
abs_error_4_3 = np.abs(error_4_3)

e_4 = df_4['Estimated_Flow'].values
e_4_2 = dt_4['Estimated_Flow'].values
e_4_3 = d_4['Estimated_Flow'].values

# # 箱线图
# plt.figure(figsize=(9, 6))
# plt.subplot(2, 2, 1)
# plt.boxplot([abs_error_1, abs_error_1_2, abs_error_1_3], labels=['UE', 'fusion', 'lgbm'], showfliers=False)
# plt.ylabel('Absolute Error')
# plt.ylim(-1, 37.9)
# plt.title('Scenario 1')
#
# plt.subplot(2, 2, 2)
# plt.boxplot([abs_error_2_2, abs_error_2_3, abs_error_2], labels=['UE', 'fusion', 'lgbm'], showfliers=False)
# plt.ylabel('Absolute Error')
# plt.ylim(-1, 37.9)
# plt.title('Scenario 2')
#
# plt.subplot(2, 2, 3)
# plt.boxplot([abs_error_3, abs_error_3_2, abs_error_3_3], labels=['UE', 'fusion', 'lgbm'], showfliers=False)
# plt.ylabel('Absolute Error')
# plt.ylim(-1, 37.9)
# plt.title('Scenario 3')
#
# plt.subplot(2, 2, 4)
# plt.boxplot([abs_error_4, abs_error_4_2, abs_error_4_3], labels=['UE', 'fusion', 'lgbm'], showfliers=False)
# plt.ylabel('Absolute Error')
# plt.ylim(-1, 37.9)
# plt.title('Scenario 4')
#
# plt.tight_layout()
# plt.show()

# # 误差与流量关系
# plt.figure(figsize=(10, 7))
# plt.subplot(2, 2, 1)
# plt.scatter(t_1, abs_error_1_2, alpha=0.7, label='UE', marker='^', color='#fe7f0e')
# plt.scatter(t_1, abs_error_1, alpha=0.9, label='Fusion', color='#274b83')
# plt.scatter(t_1, abs_error_1_3, alpha=0.9, label='lgbm', marker='*', color='#efbed3')
# plt.xlabel('True Flow')
# plt.ylabel('Absolute Error')
# plt.title('Scenario 1')
# plt.legend()
#
# plt.subplot(2, 2, 2)
# plt.scatter(t_1, abs_error_2_2, alpha=0.7, label='UE', marker='^', color='#fe7f0e')
# plt.scatter(t_1, abs_error_2, alpha=0.9, label='Fusion', color='#274b83')
# plt.scatter(t_1, abs_error_2_3, alpha=0.9, label='lgbm', marker='*', color='#efbed3')
# plt.xlabel('True Flow')
# plt.ylabel('Absolute Error')
# plt.title('Scenario 2')
# plt.legend()
#
# plt.subplot(2, 2, 3)
# plt.scatter(t_1, abs_error_3_2, alpha=0.7, label='UE', marker='^', color='#fe7f0e')
# plt.scatter(t_1, abs_error_3, alpha=0.9, label='Fusion', color='#274b83')
# plt.scatter(t_1, abs_error_3_3, alpha=0.9, label='lgbm', marker='*', color='#efbed3')
# plt.xlabel('True Flow')
# plt.ylabel('Absolute Error')
# plt.title('Scenario 3')
# plt.legend()
#
# plt.subplot(2, 2, 4)
# plt.scatter(t_1, abs_error_4_2, alpha=0.7, label='UE', marker='^', color='#fe7f0e')
# plt.scatter(t_1, abs_error_4, alpha=0.9, label='Fusion', color='#274b83')
# plt.scatter(t_1, abs_error_4_3, alpha=0.9, label='lgbm', marker='*', color='#efbed3')
# plt.xlabel('True Flow')
# plt.ylabel('Absolute Error')
# plt.title('Scenario 4')
# plt.legend()
#
# plt.tight_layout()
# plt.show()


# # CDF曲线
# def plot_cdf(data):
#     sorted_data = np.sort(data)
#     yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
#     return sorted_data, yvals
#
#
# x_1, y_1 = plot_cdf(abs_error_1)
# x_1_2, y_1_2 = plot_cdf(abs_error_1_2)
# x_1_3, y_1_3 = plot_cdf(abs_error_1_3)
#
# x_2, y_2 = plot_cdf(abs_error_2)
# x_2_2, y_2_2 = plot_cdf(abs_error_2_2)
# x_2_3, y_2_3 = plot_cdf(abs_error_2_3)
#
# x_3, y_3 = plot_cdf(abs_error_3)
# x_3_2, y_3_2 = plot_cdf(abs_error_3_2)
# x_3_3, y_3_3 = plot_cdf(abs_error_3_3)
#
# x_4, y_4 = plot_cdf(abs_error_4)
# x_4_2, y_4_2 = plot_cdf(abs_error_4_2)
# x_4_3, y_4_3 = plot_cdf(abs_error_4_3)
#
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 2, 1)
# plt.plot(x_1, y_1, label='fusion')
# plt.plot(x_1_2, y_1_2, label='UE')
# plt.plot(x_1_3, y_1_3, label='lgbm')
#
# plt.xlabel('Absolute Error')
# plt.ylabel('CDF')
# plt.title('Scenario 1')
#
# plt.grid(linestyle='dashed', alpha=0.8)
# plt.legend()
#
# plt.subplot(2, 2, 2)
# plt.plot(x_2, y_2, label='fusion')
# plt.plot(x_2_2, y_2_2, label='UE')
# plt.plot(x_2_3, y_2_3, label='lgbm')
#
# plt.xlabel('Absolute Error')
# plt.ylabel('CDF')
# plt.title('Scenario 2')
#
# plt.grid(linestyle='dashed', alpha=0.8)
# plt.legend()
#
# plt.subplot(2, 2, 3)
# plt.plot(x_3, y_3, label='fusion')
# plt.plot(x_3_2, y_3_2, label='UE')
# plt.plot(x_3_3, y_3_3, label='lgbm')
#
# plt.xlabel('Absolute Error')
# plt.ylabel('CDF')
# plt.title('Scenario 3')
#
# plt.grid(linestyle='dashed', alpha=0.8)
# plt.legend()
#
# plt.subplot(2, 2, 4)
# plt.plot(x_4, y_4, label='fusion')
# plt.plot(x_4_2, y_4_2, label='UE')
# plt.plot(x_4_3, y_4_3, label='lgbm')
#
# plt.xlabel('Absolute Error')
# plt.ylabel('CDF')
# plt.title('Scenario 4')
#
# plt.grid(linestyle='dashed', alpha=0.8)
# plt.legend()
#
# plt.tight_layout()
# plt.show()

# # 散点图
# plt.figure(figsize=(10, 7))
# plt.subplot(2, 2, 1)
# plt.scatter(t_1, e_1_2, alpha=0.7, label='UE', marker='^', color='#fe7f0e')
# plt.scatter(t_1, e_1, alpha=0.9, label='Fusion', color='#274b83')
# plt.scatter(t_1, e_1_3, alpha=0.9, label='lgbm', marker='*', color='#efbed3')
# # 45度参考线
# min_val = min(t_1.min(), e_1.min())
# max_val = max(t_1.max(), e_1.max())
# plt.plot([min_val, max_val], [min_val, max_val], color='black', alpha=0.5, label='45° line')
# plt.xlabel('True Flow')
# plt.ylabel('Estimated Flow')
# plt.title('Scenario 1')
# plt.legend()
#
# plt.subplot(2, 2, 2)
# plt.scatter(t_1, e_2_2, alpha=0.7, label='UE', marker='^', color='#fe7f0e')
# plt.scatter(t_1, e_2, alpha=0.9, label='Fusion', color='#274b83')
# plt.scatter(t_1, e_2_3, alpha=0.9, label='lgbm', marker='*', color='#efbed3')
# # 45度参考线
# min_val = min(t_1.min(), e_2.min())
# max_val = max(t_1.max(), e_2.max())
# plt.plot([min_val, max_val], [min_val, max_val], color='black', alpha=0.5, label='45° line')
# plt.xlabel('True Flow')
# plt.ylabel('Estimated Flow')
# plt.title('Scenario 2')
# plt.legend()
#
# plt.subplot(2, 2, 3)
# plt.scatter(t_1, e_3_2, alpha=0.7, label='UE', marker='^', color='#fe7f0e')
# plt.scatter(t_1, e_3, alpha=0.9, label='Fusion', color='#274b83')
# plt.scatter(t_1, e_3_3, alpha=0.9, label='lgbm', marker='*', color='#efbed3')
# # 45度参考线
# min_val = min(t_1.min(), e_3.min())
# max_val = max(t_1.max(), e_3.max())
# plt.plot([min_val, max_val], [min_val, max_val], color='black', alpha=0.5, label='45° line')
# plt.xlabel('True Flow')
# plt.ylabel('Estimated Flow')
# plt.title('Scenario 3')
# plt.legend()
#
# plt.subplot(2, 2, 4)
# plt.scatter(t_1, e_4_2, alpha=0.7, label='UE', marker='^', color='#fe7f0e')
# plt.scatter(t_1, e_4, alpha=0.9, label='Fusion', color='#274b83')
# plt.scatter(t_1, e_4_3, alpha=0.9, label='lgbm', marker='*', color='#efbed3')
# # 45度参考线
# min_val = min(t_1.min(), e_4.min())
# max_val = max(t_1.max(), e_4.max())
# plt.plot([min_val, max_val], [min_val, max_val], color='black', alpha=0.5, label='45° line')
# plt.xlabel('True Flow')
# plt.ylabel('Estimated Flow')
# plt.title('Scenario 4')
# plt.legend()
#
# plt.tight_layout()
# plt.show()

# sorted_idx = np.argsort(true_flow)
#
# plt.figure()
#
# plt.plot(true_flow[sorted_idx], label='True')
# plt.plot(pred_flow[sorted_idx], label='Estimated')
#
# plt.legend()
# plt.title('Flow Comparison (Sorted by True Flow)')
#
# plt.tight_layout()
# plt.show()