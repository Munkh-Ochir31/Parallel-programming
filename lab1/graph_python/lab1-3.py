import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


sequential_path = './csv/result_sequential_matrix.csv'
threads_path    = './csv/result_threads_matrix.csv'
openmp_path     = './csv/result_openmp_matrix.csv'

sequential_battery_path = './csv/result_sequential_matrix_battery.csv'
threads_battery_path    = './csv/result_threads_matrix_battery.csv'
openmp_battery_path     = './csv/result_openmp_matrix_battery.csv'


def mean_time_by_threads(df, thread_col=2, time_col=5):
    out = (df.groupby(df.columns[thread_col], as_index=False)[df.columns[time_col]]
             .mean()
             .rename(columns={df.columns[thread_col]:'threads', df.columns[time_col]:'mean_time'})
             .sort_values('threads'))
    return out

def speed_eff(t1, mean_df):
    mean_df = mean_df.copy()
    mean_df['speedup'] = t1 / mean_df['mean_time']
    mean_df['eff'] = mean_df['speedup'] / mean_df['threads']
    return mean_df

csv_seq_plug = pd.read_csv(sequential_path)
csv_thr_plug = pd.read_csv(threads_path)
csv_omp_plug = pd.read_csv(openmp_path)

csv_seq_bat  = pd.read_csv(sequential_battery_path)
csv_thr_bat  = pd.read_csv(threads_battery_path)
csv_omp_bat  = pd.read_csv(openmp_battery_path)

t1_plugged = csv_seq_plug.iloc[:, 5].mean()
t1_battery = csv_seq_bat.iloc[:, 5].mean()

thr_plug = mean_time_by_threads(csv_thr_plug)
omp_plug = mean_time_by_threads(csv_omp_plug)

thr_bat  = mean_time_by_threads(csv_thr_bat)
omp_bat  = mean_time_by_threads(csv_omp_bat)

thr_plug = speed_eff(t1_plugged, thr_plug)
omp_plug = speed_eff(t1_plugged, omp_plug)

thr_bat  = speed_eff(t1_battery, thr_bat)
omp_bat  = speed_eff(t1_battery, omp_bat)

all_threads = np.array(sorted(set(thr_plug['threads']) | set(omp_plug['threads'])
                              | set(thr_bat['threads'])  | set(omp_bat['threads'])))
ideal_speed = all_threads
ideal_eff   = np.ones_like(all_threads, dtype=float)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

c_thr_bat = '#b22222'   
c_omp_bat = '#ff6f6f' 
c_thr_ac  = '#0b6623'
c_omp_ac  = '#66c266'
c_ideal   = 'gray'

lw = 2
ms = 7

ax1.plot(thr_bat['threads'],  thr_bat['speedup'], marker='o', linestyle='--',
         color=c_thr_bat, linewidth=lw, markersize=ms, label='threads (Battery)')
ax1.plot(omp_bat['threads'],  omp_bat['speedup'], marker='o', linestyle='--',
         color=c_omp_bat, linewidth=lw, markersize=ms, label='OpenMP (Battery)')
ax1.plot(thr_plug['threads'], thr_plug['speedup'], marker='o', linestyle='-',
         color=c_thr_ac, linewidth=lw, markersize=ms, label='threads (Plugged-in)')
ax1.plot(omp_plug['threads'], omp_plug['speedup'], marker='o', linestyle='-',
         color=c_omp_ac, linewidth=lw, markersize=ms, label='OpenMP (Plugged-in)')
ax1.plot(all_threads, ideal_speed, linestyle='--', color=c_ideal, linewidth=lw, label='Ideal (S=T)')

ax1.set_title('Matrix Multiplication: Speed (Battery vs Plugged-in)', fontsize=14)
ax1.set_ylabel('Speed (T1 / Tn)', fontsize=12)
ax1.set_xlim(0.5, all_threads[-1] + 0.5)
ax1.set_xticks(all_threads)
ax1.grid(True, linestyle=':', linewidth=1, alpha=0.6)
ax1.set_axisbelow(True)

for x in (4, 8):
    ax1.axvline(x, color=c_ideal, linestyle=':', linewidth=1.5, alpha=0.8)

ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

ax2.plot(thr_bat['threads'],  thr_bat['eff'], marker='o', linestyle='--',
         color=c_thr_bat, linewidth=lw, markersize=ms, label='threads (Battery)')
ax2.plot(omp_bat['threads'],  omp_bat['eff'], marker='o', linestyle='--',
         color=c_omp_bat, linewidth=lw, markersize=ms, label='OpenMP (Battery)')
ax2.plot(thr_plug['threads'], thr_plug['eff'], marker='o', linestyle='-',
         color=c_thr_ac, linewidth=lw, markersize=ms, label='threads (Plugged-in)')
ax2.plot(omp_plug['threads'], omp_plug['eff'], marker='o', linestyle='-',
         color=c_omp_ac, linewidth=lw, markersize=ms, label='OpenMP (Plugged-in)')
ax2.plot(all_threads, ideal_eff, linestyle='--', color=c_ideal, linewidth=lw, label='Ideal (E=1)')

ax2.set_title('Matrix Multiplication: Efficiency (Battery vs Plugged-in)', fontsize=14)
ax2.set_xlabel(f'Threads (1–{int(all_threads[-1])})', fontsize=12)
ax2.set_ylabel('Efficiency', fontsize=12)
ax2.set_ylim(0, 1.02)
ax2.set_xticks(all_threads)
ax2.grid(True, linestyle=':', linewidth=1, alpha=0.6)
ax2.set_axisbelow(True)

for x in (4, 8):
    ax2.axvline(x, color=c_ideal, linestyle=':', linewidth=1.5, alpha=0.8)

ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

left_note = (
    "Battery үед CPU power/давтамж ихэвчлэн хязгаарлагддаг тул ханалт эрт үүсч болно.\n"
    "Plugged-in үед давтамж тогтвортой тул Speed/Efficiency илүү сайн гарах хандлагатай."
)

cpu_info = (
    "CPU: Intel(R) Core(TM) i7-13700H\n"
    "Cores / Threads: 14 / 20\n"
    "L3 Cache: 24 MiB\n"
    "Matrix Multiplication (with transpose)"
)

plt.tight_layout(rect=[0.03, 0.22, 0.86, 0.98])

fig.text(0.08, 0.08, left_note, ha='left',  va='bottom', fontsize=11)
fig.text(0.92, 0.08, cpu_info,  ha='right', va='bottom', fontsize=11)
plt.show()
