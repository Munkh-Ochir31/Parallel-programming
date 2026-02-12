import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

no_transpose_df = pd.read_csv('./csv/result_openmp_mxn_no_transpose.csv')
with_transpose_df = pd.read_csv('./csv/result_openmp_mxn_with_transpose.csv')

def mean_time_by_threads(df):
    return df.groupby('threads')['elapsed_ms'].mean()

no_transpose_mean = mean_time_by_threads(no_transpose_df)
with_transpose_mean = mean_time_by_threads(with_transpose_df)

t1_no_transpose = no_transpose_mean[1]
t1_with_transpose = with_transpose_mean[1]

def calculate_metrics(mean_times, t1):
    threads = mean_times.index.values
    speedup = t1 / mean_times.values
    efficiency = speedup / threads
    return threads, speedup, efficiency

threads_nt, speedup_nt, efficiency_nt = calculate_metrics(no_transpose_mean, t1_no_transpose)
threads_t, speedup_t, efficiency_t = calculate_metrics(with_transpose_mean, t1_with_transpose)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Matrix Multiplication', fontsize=16, fontweight='bold')

ax1 = axes[0]
ax1.plot(threads_nt, no_transpose_mean.values, 'o-', label='No Transpose', linewidth=2, markersize=8, color='#e74c3c')
ax1.plot(threads_t, with_transpose_mean.values, 's-', label='With Transpose', linewidth=2, markersize=8, color='#27ae60')
ax1.set_xlabel('Thread Count', fontsize=11)
ax1.set_ylabel('Execution Time (ms)', fontsize=11)
ax1.set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(threads_nt)

ax2 = axes[1]
ax2.plot(threads_nt, speedup_nt, 'o-', label='No Transpose', linewidth=2, markersize=8, color='#e74c3c')
ax2.plot(threads_t, speedup_t, 's-', label='With Transpose', linewidth=2, markersize=8, color='#27ae60')

ax2.plot(threads_nt, threads_nt, '--', label='Ideal Speedup', linewidth=2, color='gray', alpha=0.7)
ax2.set_xlabel('Thread Count', fontsize=11)
ax2.set_ylabel('Speedup', fontsize=11)
ax2.set_title('Speedup Comparison', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(threads_nt)

ax3 = axes[2]
ax3.plot(threads_nt, efficiency_nt, 'o-', label='No Transpose', linewidth=2, markersize=8, color='#e74c3c')
ax3.plot(threads_t, efficiency_t, 's-', label='With Transpose', linewidth=2, markersize=8, color='#27ae60')

ax3.axhline(y=1.0, linestyle='--', label='Ideal Efficiency (100%)', linewidth=2, color='gray', alpha=0.7)
ax3.set_xlabel('Thread Count', fontsize=11)
ax3.set_ylabel('Efficiency', fontsize=11)
ax3.set_title('Efficiency Comparison', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(threads_nt)
ax3.set_ylim([0, 1.2])

plt.tight_layout()
plt.show()
