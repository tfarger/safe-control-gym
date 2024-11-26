import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define the base directory
base_dir = 'examples/hpo/hpo'  # Change this if needed
algorithms = ['pid', 'lqr', 'ilqr', 'linear_mpc', 'mpc_acados', 'ppo', 'sac', 'dppo']  # List your algorithms here
trials = 40  # Number of trials for HPO

# Function to get the seed folder with the smallest seed number


def get_smallest_seed_folder(algorithm, package):
    package_dir = os.path.join(base_dir, algorithm, package)
    seed_folders = [f for f in os.listdir(package_dir) if os.path.isdir(os.path.join(package_dir, f)) and f.startswith('seed')]

    # Extract the seed numbers and sort to find the smallest
    seed_numbers = [(folder, int(folder.split('_')[0][4:])) for folder in seed_folders]
    smallest_seed_folder = min(seed_numbers, key=lambda x: x[1])[0] if seed_numbers else None

    return smallest_seed_folder

# Function to load trials data from the smallest seed folder


def load_trials_data(algorithm, package):
    seed_folder = get_smallest_seed_folder(algorithm, package)
    if seed_folder:
        file_path = os.path.join(base_dir, algorithm, package, seed_folder, 'hpo', 'trials.csv')
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
    return None

# Function to load hand-tuned performance data


def load_handtune_data(algorithm):
    seed_folder = get_smallest_seed_folder(algorithm, 'optuna')
    file_path = os.path.join(base_dir, algorithm, 'optuna', seed_folder, 'hpo', 'warmstart_trial_value.txt')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            content = file.read().strip('[]')
            numbers = [float(x) for x in content.split(',')]
            return np.mean(numbers)
    return None

# Plot Hyperparameter Evaluation Over Trials


def plot_hpo_evaluation(trials, algorithms, packages=['optuna', 'vizier'], target_name='Normalized Reward'):
    plt.style.use('ggplot')  # Use ggplot style for consistent appearance
    num_algorithms = len(algorithms)
    num_packages = len(packages)
    fig, axes = plt.subplots(num_packages, num_algorithms, figsize=(40, 10), sharex=True, sharey=True)
    fig.suptitle('Optimization History Plot for Hyperparameter Tuning', fontsize=16)
    cmap = plt.get_cmap('tab10')  # Colormap for distinguishing algorithms

    for row, package in enumerate(packages):
        for col, algorithm in enumerate(algorithms):
            ax = axes[row, col]
            ax.set_title(f'{package} - {algorithm}')
            ax.set_xlabel('Trial Number')
            ax.set_ylabel(target_name)

            # Load trial data
            trials_data = load_trials_data(algorithm, package)
            if trials_data is not None:
                trial_numbers = trials_data['number'][:trials]
                values = trials_data['value'][:trials]

                # Scatter plot for individual trials
                ax.scatter(trial_numbers, values, color=cmap(col), alpha=0.8, label=f'{package} - {algorithm}')

                # Optionally, plot best values as cumulative maximum
                best_values = np.maximum.accumulate(values)
                ax.plot(trial_numbers, best_values, color=cmap(col), alpha=0.6, linestyle='--', label='Best Values')

            ax.legend()

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('hpo_optimization.png')

# Box Plot Comparison of Performance


def plot_performance_comparison(algorithms, packages=['optuna', 'vizier']):
    data = []
    data_rmse = []

    for algorithm in algorithms:
        # Hand-tuned data
        handtune_value = load_handtune_data(algorithm)
        if handtune_value is not None:
            data.append([algorithm, 'Hand Tuned', handtune_value])
            data_rmse.append([algorithm, 'Hand Tuned', -np.log(handtune_value)])

        # HPO data for each package
        for package in packages:
            trials_data = load_trials_data(algorithm, package)
            if trials_data is not None:
                data.append([algorithm, package, trials_data['value'].max()])
                rmse = -np.log(trials_data['value'].max())
                data_rmse.append([algorithm, package, rmse])

    # Convert to DataFrame for plotting
    df = pd.DataFrame(data, columns=['Algorithm', 'Method', 'Normalized Reward'])

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Algorithm', y='Normalized Reward', hue='Method', data=df)
    sns.move_legend(plt.gca(), 'lower left')
    plt.title('Performance Comparison of Hand-Tuned, Optuna, and Vizier')
    plt.savefig('hpo_performance_comparison.png')

    df_rmse = pd.DataFrame(data_rmse, columns=['Algorithm', 'Method', 'RMSE'])
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Algorithm', y='RMSE', hue='Method', data=df_rmse)
    sns.move_legend(plt.gca(), 'lower left')
    plt.title('RMSE Comparison of Hand-Tuned, Optuna, and Vizier')
    plt.savefig('hpo_rmse_comparison.png')


# Plot evaluation over trials for each algorithm
plot_hpo_evaluation(trials, algorithms)

# Plot box plot comparison across algorithms
plot_performance_comparison(algorithms)
