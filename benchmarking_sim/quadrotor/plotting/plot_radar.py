import os
import sys

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

script_dir = os.path.dirname(__file__)

# get the pyplot default color wheel
prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# plot_colors = {
#     'GP-MPC': colors[0],
#     'PPO': colors[1],
#     'SAC': colors[3],
#     # 'iLQR': 'darkgray',
#     'DPPO': colors[6],
#     'Linear-MPC': colors[2],
#     'MPC': colors[-1],
#     'MAX': 'none',
#     'MIN': 'none',
# }

plot_colors = {
    'GP-MPC': 'royalblue',
    'PPO': 'darkorange',
    'SAC': 'red',
    'DPPO': 'tab:pink',
    'iLQR': 'darkgray',
    'Linear MPC': 'green',
    'Nonlinear MPC': 'cadetblue',
    'MAX': 'none',
    'MIN': 'none',
    'FMPC': 'purple'
}

axis_label_fontsize = 30
text_fontsize = 30
supertitle_fontsize = 30
subtitle_fontsize = 30
small_text_size = 20


def spider(df, *, id_column, title=None, subtitle=None, max_values=None, padding=1.25, plt_name=''):
    categories = df._get_numeric_data().columns.tolist()
    data = df[categories].to_dict(orient='list')
    ids = df[id_column].tolist()

    lower_padding = (padding - 1) / 2
    # upper_padding = 1 + lower_padding * 2
    upper_padding = 1 + 7 * lower_padding
    # upper_padding = 1.05
    flip_flag = True

    if max_values is None:
        max_values = {key: upper_padding * max(value) for key, value in data.items()}

    normalized_data = {key: np.array(value) / max_values[key] + lower_padding for key, value in data.items()}
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    print('tiks:', tiks)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True), )
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]

        # Invert the values to have the higher values in the center
        values[0:5] = 1 - np.array(values[0:5])

        values += values[:1]  # Close the plot for a better look
        # values = 1 - np.array(values) 
        if model_name in ['MAX', 'MIN']:
            ax.plot(angles, values, color=plot_colors[model_name], )
            ax.scatter(angles, values, facecolor=plot_colors[model_name], )
            ax.fill(angles, values, alpha=0.15, color=plot_colors[model_name], )
            continue
        else:
            ax.plot(angles, values, label=model_name, color=plot_colors[model_name], )
            ax.scatter(angles, values, facecolor=plot_colors[model_name], )
            ax.fill(angles, values, alpha=0.15, color=plot_colors[model_name], )
        for _x, _y, t in zip(angles, values, actual_values):
            if _x == angles[2]:
                t = f'{t:.4f}' if isinstance(t, float) else str(t)
            elif _x == angles[4]:  # sampling complexity
                if t == int(1):
                    # _y = 0.01
                    t = '0'
                else:
                    # write number in 1e5 format
                    t = f'{t:.1E}'  # if isinstance(t, float) else str(t)
            # elif _x == angles[0]:
            #     t = f'{t:.2f}' if isinstance(t, float) else str(t)
            else:
                t = f'{t:.3f}' if isinstance(t, float) else str(t)
            if t == '1': t = 'Model-free'
            if t == '40': t = '   Linear\n   model'
            if t == '80': t = 'Nonlinear\n   model'

            t = t.center(10, ' ')
            if _x == angles[3]:
                ax.text(_x + 0.2, _y + 0.15, t, size=small_text_size)
            elif _x == angles[0]:
                if flip_flag:
                    ax.text(_x + 0.2, _y - 0.1, t, size=small_text_size)
                else:
                    ax.text(_x - 0.2, _y + 0.0, t, size=small_text_size)
                flip_flag = False
            elif model_name == 'GP-MPC':
                if _x == angles[0]:
                    if flip_flag:
                        ax.text(_x + 0.05, _y - 0.1, t, size=small_text_size)
                    else:
                        ax.text(_x - 0.05, _y, t, size=small_text_size)
                    flip_flag = False
                elif _x == angles[4]:
                    ax.text(_x, _y - 0.05, t, size=small_text_size)
                else:
                    ax.text(_x, _y - 0.01, t, size=small_text_size)

            elif model_name == 'DPPO':
                # if _x == angles[5]:
                #     ax.text(_x, _y-0.1, t, size=small_text_size)
                # if _x == angles[0]: # generalization performance
                #     ax.text(_x-0.1, _y-0.05, t, size=small_text_size)
                # elif _x == angles[2]: # inference time
                #     ax.text(_x+0.1, _y-0.05, t, size=small_text_size)
                if _x == angles[4]:  # sampling complexity
                    ax.text(_x, _y + 0.15, t, size=small_text_size)
                else:
                    ax.text(_x, _y - 0.01, t, size=small_text_size)
            else:
                ax.text(_x, _y - 0.01, t, size=small_text_size)

    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05, color='lightgray')
    # ax.fill(angles[0:3], np.ones(3), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks, fontsize=axis_label_fontsize)
    # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.2), fontsize=text_fontsize)
    if title is not None: plt.suptitle(title, fontsize=supertitle_fontsize)
    if subtitle is not None: plt.title(subtitle, fontsize=subtitle_fontsize)
    plt.show()
    fig_save_path = os.path.join(script_dir, f'{plt_name}_radar.pdf')
    fig.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    print(f'figure saved as {fig_save_path}')


radar = spider
    
def run():

    radar = spider

    num_axis = 6
    gen_performance = [0.07240253210609013, 0.031425261835490235,  # GP-MPC
                    0.10363015782869267, 0.03574250675445726,  # Linear-MPC
                    0.06669989755434953, 0.023313325828377546,  # MPC
                    0.049631400535807925, 0.047894774207991445,  # PPO
                    0.12754716185055215, 0.08743083392874743,  # SAC
                    0.050426782816096215, 0.04861203156962396,  # DPPO
                    ]
    performance = [0.049872511450839645, 0.049872511450839645,  # GP-MPC
                0.06556989411791102, 0.06556989411791102,  # Linear-MPC
                0.04421752503119518, 0.04421752503119518,  # MPC
                0.01746153113470009, 0.01746153113470009,  # PPO
                0.0764178745409007, 0.0764178745409007,  # SAC
                0.01931011838369746, 0.01931011838369746,  # DPPO
                ]
    inference_time = [0.0090775150246974736, 0.0090775150246974736,
                    0.0011251235, 0.0011251235,
                    0.0061547613, 0.0061547613,
                    0.00020738168999000832, 0.00020738168999000832,
                    0.00024354409288477016, 0.00024354409288477016,
                    0.0001976909460844817, 0.0001976909460844817,
                    ]
    model_complexity = [80, 80,
                        40, 40,
                        80, 80,
                        1, 1,
                        1, 1,
                        1, 1]
    sampling_complexity = [int(660), int(660),
                        int(1), int(1),
                        int(1), int(1),
                        int(2.8 * 1e5), int(2.8 * 1e5),
                        int(2 * 1e5), int(2 * 1e5),
                        int(2.5 * 1e5), int(2.5 * 1e5)]
    robustness = [120, 120,
                90, 90,
                90, 90,
                10, 10,
                30, 30,
                20, 20]
    data = [gen_performance, performance, inference_time, model_complexity, sampling_complexity, robustness]
    max_values = [0.01, 0.01, 1e-5, 1, 1, 120]
    min_values = [0.2, 0.2, 1e-2, 80, 3.e5, 1]

    for i, d in enumerate(data):
        data[i].append(max_values[i])
        data[i].append(min_values[i])

    # append the max and min values to the data
    algos = ['GP-MPC', 'GP-MPC',
            'Linear MPC', 'Linear MPC',
            'Nonlinear MPC', 'Nonlinear MPC',
            'PPO', 'PPO',
            'SAC', 'SAC',
            'DPPO', 'DPPO',
            'MAX', 'MIN']
        # apppend the max and min values to the data

    # read the argv
    if len(sys.argv) > 1:
        masks_algo = [int(i) for i in sys.argv[1:]]
        masks_algo.append(6)
        masks_algo.append(7)
    else:
        masks_algo = [8, 9, -2, -1]
    data = np.array(data)[:, masks_algo]
    data = data.tolist()
    algos = [algos[i] for i in masks_algo]
    print(algos)

    spider(
        pd.DataFrame({
            # 'x': [*'ab'],
            'x': algos,
            '$\qquad\qquad\qquad\quad$  Generalization\n $\qquad\qquad\qquad\quad$ performance\n\n':
                data[0],
            '$\qquad\qquad\qquad\quad$ Performance\n':
                data[1],
            # '$\quad\quad\quad\quad\quad\qquad$(Figure-8 tracking)': [3.94646538e-02, 0.03],
            'Inference\ntime\n\n':
                data[2],
            'Model                \nknowledge                ':
                [int(data[3][i]) for i in range(len(data[3]))],
            '\n\n\nSampling\ncomplexity':
                data[4],
            '\n\nRobustness':
                [int(data[5][i]) for i in range(len(data[5]))],
        }),

        id_column='x',
        # title='   Overall Comparison',
        # title = algos[0],
        title=None,
        # subtitle='(Normalized linear scale)',
        padding=1.1,
        # padding=1,
        plt_name=algos[0],
    )

if __name__ == '__main__':
    run()