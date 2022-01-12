import matplotlib.pyplot as plt
from collections import defaultdict
from tests.test_monotone import test_monotone_memreps
from tests.test_dfa_learning import test_gridworld_dfa, test_tomita
from pprint import pprint
import numpy as np
import pickle

def run_experiment(cost_ratios, number_of_trials, plotfilename, outfilename, monotone_exp=True, tomita=-1):
    results = {}
    membership_results = {}
    for membership_cost in cost_ratios:
        print("Collecting results for membership cost".format(membership_cost))
        membership_results[membership_cost] = defaultdict(list)
        results[membership_cost] = defaultdict(list)
        for trial in range(number_of_trials):
            print("============On trial {}==============".format(trial))
            res, dim = np.random.randint(4, 7), 2
            random_concept_params = np.random.randint(1, res, dim)
            if monotone_exp:
                counts = test_monotone_memreps(membership_cost=membership_cost,
                                               initial_point=random_concept_params, res=res)
            else:
                if tomita == -1:
                    counts = test_gridworld_dfa(membership_cost)
                else:
                    counts = test_tomita(tomita, membership_cost)
            results[membership_cost]['pref'].append(counts['≺'])
            results[membership_cost]['mem'].append(counts['∈'])
            results[membership_cost]['equiv'].append(counts['≡'])

            if monotone_exp:
                counts_membership = test_monotone_memreps(membership_cost=membership_cost,
                                                          initial_point=random_concept_params, res=res, force_membership=True)
            else:
                if tomita == -1:
                    counts_membership = test_gridworld_dfa(membership_cost=membership_cost, force_membership=True)
                else:
                    counts_membership = test_tomita(tomita, membership_cost=membership_cost, force_membership=True)
            membership_results[membership_cost]['pref'].append(counts_membership['≺'])
            membership_results[membership_cost]['mem'].append(counts_membership['∈'])
            membership_results[membership_cost]['equiv'].append(counts_membership['≡'])

    with open(outfilename, 'wb') as outfile:
        pickle.dump({"results" : results, "membership_results" : membership_results}, outfile)
    plot_bars(results, membership_results, plotfilename)

def plot_bars(results_dict, baseline_dict, plotfilename):
    fig = plt.figure()
    num_ratios = len(results_dict)
    columns = []
    res_data = np.zeros((3, num_ratios))
    res_stds = np.zeros((3, num_ratios))
    base_data = np.zeros((3, num_ratios))
    base_stds = np.zeros((3, num_ratios))
    membership_queries = []
    total_queries = []
    baseline_membership = []
    for ratio_idx, (ratio, counts) in enumerate(results_dict.items()):
        columns.append(ratio)
        avg_membership, std_membership = np.mean(counts['mem']), np.std(counts['mem'])
        avg_preference, std_preference = np.mean(counts['pref']), np.std(counts['pref'])
        avg_equivalence, std_equivalence = np.mean(counts['equiv']), np.std(counts['equiv'])
        res_data[:, ratio_idx] = [avg_membership, avg_preference, avg_equivalence]
        res_stds[:, ratio_idx] = [std_membership, std_preference, std_equivalence]
        membership_queries.append(avg_membership)
        total_queries.append(avg_membership + avg_preference + avg_equivalence)
    for ratio_idx, (ratio, counts) in enumerate(baseline_dict.items()):
        avg_base_membership, std_membership = np.mean(counts['mem']), np.std(counts['mem'])
        avg_preference, std_preference = np.mean(counts['pref']), np.std(counts['pref'])
        avg_equivalence, std_equivalence = np.mean(counts['equiv']), np.std(counts['equiv'])
        baseline_membership.append(avg_base_membership)
        base_data[:, ratio_idx] = [avg_base_membership, avg_preference, avg_equivalence]
        base_stds[:, ratio_idx] = [std_membership, std_preference, std_equivalence]
    cmap = plt.cm.get_cmap('Pastel1')
    X_axis = np.arange(num_ratios)
    bar_width = 0.4
    res_y_offset = np.zeros(len(columns))
    for row in range(3):
        plt.bar(X_axis, res_data[row], bar_width,# yerr=res_stds[row],
                capsize=3, color=cmap(row), bottom=res_y_offset, edgecolor="black")
        res_y_offset = res_y_offset + res_data[row]
    print("Results: avg queries, std on queries.")
    print("M+P: ", res_data, res_stds)
    print("M ONLY: ", base_data, base_stds)
    plt.plot(X_axis, [np.mean(baseline_membership)] * num_ratios, color='red', label="Membership-only baseline", linestyle='dashed', linewidth = 2, markersize = 12)
    plt.plot(X_axis, membership_queries, color='purple', label="# of membership queries used", linestyle='dashed', linewidth = 2, markersize = 12)
    plt.plot(X_axis, total_queries, color='green', linestyle='dashed', label="# of total queries used", linewidth = 2, markersize = 12)
    plt.xticks(X_axis, columns)
    plt.legend()
    plt.title('Trading off membership queries for preference queries')
    plt.ylabel('Mean number of queries')
    plt.xlabel('Ratio of membership cost to preference cost')
    plt.savefig(plotfilename, format='pdf')


run_experiment([1,2,4,8], 50, "original_dfa_plots_monotone.pdf", "original_dfa_raws_monotone", monotone_exp=True, tomita=6)
