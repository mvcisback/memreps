import matplotlib.pyplot as plt
from collections import defaultdict
from tests.test_monotone import test_monotone_memreps
import numpy as np

def run_experiment(cost_ratios, number_of_trials, plotfilename, outfilename):
    monotone_results = {}
    monotone_membership_results = {}
    for membership_cost in cost_ratios:
        monotone_membership_results[membership_cost] = defaultdict(list)
        monotone_results[membership_cost] = defaultdict(list)
        for trial in range(number_of_trials):
            counts = test_monotone_memreps(membership_cost)
            monotone_results[membership_cost]['pref'].append(counts['≺'])
            monotone_results[membership_cost]['mem'].append(counts['∈'])
            monotone_results[membership_cost]['equiv'].append(counts['≡'])
            counts_membership = test_monotone_memreps(membership_cost, force_membership=True)
            monotone_membership_results[membership_cost]['pref'].append(counts_membership['≺'])
            monotone_membership_results[membership_cost]['mem'].append(counts_membership['∈'])
            monotone_membership_results[membership_cost]['equiv'].append(counts_membership['≡'])

            ## TODO: add DFA experiments
    plot_bars(monotone_results, monotone_membership_results)

def plot_bars(results_dict, baseline_dict):
    fig = plt.figure()
    num_ratios = len(results_dict)
    columns = []
    res_data = np.zeros((3, num_ratios))
    res_stds = np.zeros((3, num_ratios))
    base_data = np.zeros((3, num_ratios))
    base_stds = np.zeros((3, num_ratios))
    for ratio_idx, (ratio, counts) in enumerate(results_dict.items()):
        columns.append(ratio)
        avg_membership, std_membership = np.mean(counts['mem']), np.std(counts['mem'])
        avg_preference, std_preference = np.mean(counts['pref']), np.std(counts['pref'])
        avg_equivalence, std_equivalence = np.mean(counts['equiv']), np.std(counts['equiv'])
        res_data[:, ratio_idx] = [avg_membership, avg_preference, avg_equivalence]
        res_stds[:, ratio_idx] = [std_membership, std_preference, std_equivalence]
    for ratio_idx, (ratio, counts) in enumerate(baseline_dict.items()):
        avg_membership, std_membership = np.mean(counts['mem']), np.std(counts['mem'])
        avg_preference, std_preference = np.mean(counts['pref']), np.std(counts['pref'])
        avg_equivalence, std_equivalence = np.mean(counts['equiv']), np.std(counts['equiv'])
        base_data[:, ratio_idx] = [avg_membership, avg_preference, avg_equivalence]
        base_stds[:, ratio_idx] = [std_membership, std_preference, std_equivalence]
    cmap = plt.cm.get_cmap('Pastel1')
    X_axis = np.arange(num_ratios)
    bar_width = 0.4
    res_y_offset = np.zeros(len(columns))
    base_y_offset = np.zeros(len(columns))
    for row in range(3):
        plt.bar(X_axis - 0.2, res_data[row], bar_width, yerr=res_stds[row], capsize=3, color=cmap(row), bottom=res_y_offset, edgecolor="black")
        res_y_offset = res_y_offset + res_data[row]
        plt.bar(X_axis + 0.2, base_data[row], bar_width, yerr=base_stds[row], capsize=3, color=cmap(row), bottom=base_y_offset, edgecolor="black")
        base_y_offset = base_y_offset + base_data[row]
        #plt.bar(X_axis, data[row], bar_width, color=cmap(row))
    plt.xticks(X_axis, columns)
    breakpoint()
    plt.show()