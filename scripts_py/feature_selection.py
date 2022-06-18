import os, copy, time, progressbar, multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from lifelines import CoxPHFitter
from joblib import Parallel, delayed

import viz_utils as viz_utils
import data_utils as data_utils

grid_num = 10
n_jobs = multiprocessing.cpu_count()
viz = True

def feat_grid_search_logrank_score(datasets, prote, intersected_protes, result_path):
    prote_id = intersected_protes.index(prote)
    gene_name = prote

    start_vals, end_vals = [], []
    for dataset in datasets:
        prote_vals = dataset['data'][:, prote_id]
        start_vals.append(np.quantile(prote_vals, 0.1))
        end_vals.append(np.quantile(prote_vals, 0.9))
    start_val = np.amax(start_vals)
    end_val = np.amin(end_vals)
    # print('start vals: ' + ' '.join(['{:.3f}'.format(val) for val in start_vals]))
    # print('end vals: ' + ' '.join(['{:.3f}'.format(val) for val in start_vals]))
    range_str = ' : [{:.3f}, {:.3f}]'.format(start_val, end_val)

    thresh_vals = np.arange(start_val, end_val + 1e-5, (end_val - start_val)/grid_num)[:grid_num]
    datasets_events_p_scores = np.zeros((len(datasets), 2, grid_num))
    for i, dataset in enumerate(datasets):
        DFS_p_scores, OS_p_scores = [], []
        for k, thresh in enumerate(thresh_vals):
            prote_vals = dataset['data'][:, prote_id]
            assignments = np.zeros_like(prote_vals, dtype=int)
            assignments[prote_vals - thresh > 0] = 1
            for j, (time_name, event_name) in enumerate([['DFS', 'recurrence'], ['OS', 'status']]):
                if np.mean(assignments) > 0 and np.mean(assignments) < 1:
                    p_val = data_utils.get_logrank_p(dataset[time_name], dataset[event_name], assignments)[0]
                    delta_rmst = data_utils.get_delta_rmst(dataset[time_name], dataset[event_name], assignments, 2)
                    p_score = -np.log10(p_val) * np.sign(-delta_rmst) # scpre > 0: favor
                    datasets_events_p_scores[i, j, k] = p_score

    if viz:
        viz_utils.grid_search_viz(datasets, datasets_events_p_scores, thresh_vals, gene_name, range_str, result_path)
        # viz_utils.prote_expression_in_subtypes_viz(datasets, prote, intersected_protes, gene_name, result_path)

    return datasets_events_p_scores

def dataset_prognosis_normalization(protes_datasets_events_p_scores, datasets):
    protes = datasets[0]['protes']
    for i, dataset in enumerate(datasets):
        for j, (time_name, event_name) in enumerate([('DFS', 'recurrence'), ('OS', 'status')]):
            times = dataset[time_name]
            events = dataset[event_name]
            sorted_ids = np.argsort(times)
            sorted_times = times[sorted_ids]
            sorted_events = events[sorted_ids]
            assignments = np.zeros_like(sorted_times)
            assignments[int(len(assignments)/2):] = 1
            p_val = data_utils.get_logrank_p(sorted_times, sorted_events, assignments)[0]
            p_score = -np.log10(p_val)

            fig, ax = plt.subplots(figsize=(5, 5))
            viz_utils.plot_km_curve_custom(sorted_times, sorted_events, assignments, 2, ax, dataset['label'])
            plt.tight_layout()
            plt.show()
            plt.close()

def consistency_counts(protes_datasets_events_steps_score):
    protes_events_steps_datasets_score = np.transpose(protes_datasets_events_steps_score, (0, 2, 3, 1))
    protes_events_steps_datasets_pass_favor = protes_events_steps_datasets_score > -np.log10(0.05)
    protes_events_steps_datasets_pass_unfavor = protes_events_steps_datasets_score < np.log10(0.05)
    protes_events_steps_pass_sum_favor = np.sum(protes_events_steps_datasets_pass_favor, axis=-1)
    protes_events_steps_pass_sum_unfavor = np.sum(protes_events_steps_datasets_pass_unfavor, axis=-1)
    protes_events_steps_pass_sum = np.amax(
        np.stack(
            [protes_events_steps_pass_sum_favor, protes_events_steps_pass_sum_unfavor], 
            axis=-1
        ), 
        axis=-1
    )
    protes_events_max_pass_sum = np.amax(protes_events_steps_pass_sum, axis=-1)
    protes_consistency_counts = np.sum(protes_events_max_pass_sum, axis=-1)
    return protes_consistency_counts

def grid_search_vals(datasets):
    datasets_protes_start_val, datasets_protes_end_val = [], []
    for dataset in datasets:
        protes_vals = dataset['data'].transpose()
        datasets_protes_start_val.append(np.quantile(protes_vals, 0.1, axis=1))
        datasets_protes_end_val.append(np.quantile(protes_vals, 0.9, axis=1))
    datasets_protes_start_val = np.stack(datasets_protes_start_val, axis=0)
    datasets_protes_end_val = np.stack(datasets_protes_end_val, axis=0)
    protes_start_val = np.amax(datasets_protes_start_val, axis=0)
    protes_end_val = np.amin(datasets_protes_end_val, axis=0)
    return protes_start_val, protes_end_val

def mean_score_on_mutual_significant_interval(protes_start_val, protes_end_val, protes_datasets_events_steps_score):
    protes_events_steps_datasets_score = np.transpose(protes_datasets_events_steps_score, (0, 2, 3, 1))
    protes_events_steps_datasets_pass_favor = protes_events_steps_datasets_score > -np.log10(0.05)
    protes_events_steps_datasets_pass_unfavor = protes_events_steps_datasets_score < np.log10(0.05)
    protes_events_steps_pass_sum_favor = np.sum(protes_events_steps_datasets_pass_favor, axis=-1)
    protes_events_steps_pass_sum_unfavor = np.sum(protes_events_steps_datasets_pass_unfavor, axis=-1)
    protes_events_steps_pass_sum = np.amax(
        np.stack(
            [protes_events_steps_pass_sum_favor, protes_events_steps_pass_sum_unfavor], 
            axis=-1
        ), 
        axis=-1
    )

    prote_num, event_num, step_num, dataset_num = protes_events_steps_datasets_score.shape
    protes_events_steps_all_pass = protes_events_steps_pass_sum >= dataset_num
    protes_events_steps_most_pass = protes_events_steps_pass_sum >= dataset_num - 1

    protes_events_steps_score = np.mean(protes_events_steps_datasets_score, axis=-1)

    protes_all_pass_score = np.zeros((prote_num))
    protes_most_pass_score = np.zeros((prote_num))

    for i in range(prote_num):
        events_steps_score = protes_events_steps_score[i]
        events_steps_all_pass = protes_events_steps_all_pass[i]
        events_steps_most_pass = protes_events_steps_most_pass[i]
        protes_all_pass_score[i] = np.sum(events_steps_score[events_steps_all_pass])
        protes_most_pass_score[i] = np.sum(events_steps_score[events_steps_most_pass])

    protes_grid_size = (protes_end_val - protes_start_val) / grid_num

    protes_all_pass_accum_score = protes_all_pass_score * protes_grid_size
    protes_most_pass_accum_score = protes_most_pass_score * protes_grid_size

    return protes_all_pass_accum_score, protes_most_pass_accum_score


def prognosis_consistency_selection(datasets, save_path):
    result_path = join(save_path, 'prognosis_consistency_selection')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    npy_file_path = join(result_path, 'feats_datasets_events_p_scores.npy')
    feats = datasets[0]['genes'] if 'genes' in datasets[0].keys() else datasets[0]['protes']
    if not os.path.isfile(npy_file_path):
        res = Parallel(n_jobs=n_jobs)(delayed(feat_grid_search_logrank_score)(
                datasets, feat, feats, result_path
            ) for feat in progressbar.progressbar(feats)
        )

        protes_datasets_events_p_scores = np.stack(res, axis=0)
        with open(npy_file_path, 'wb') as f:
            np.save(f, protes_datasets_events_p_scores)
    else:
        with open(npy_file_path, 'rb') as f:
            protes_datasets_events_p_scores = np.load(f)

    # dataset_prognosis_normalization(protes_datasets_events_p_scores, datasets)
    protes_consistency_counts = consistency_counts(protes_datasets_events_p_scores)

    protes_start_val, protes_end_val = grid_search_vals(datasets)

    protes_0fail_score, protes_1fail_score = mean_score_on_mutual_significant_interval(
        protes_start_val, protes_end_val, protes_datasets_events_p_scores
    )

    high_consistency_num = np.sum(protes_consistency_counts >= len(datasets) * 2 - 1)
    sorted_ids = np.argsort(-protes_consistency_counts)[:high_consistency_num]
    sorted_protes = [feats[sorted_id] for sorted_id in sorted_ids]
    sorted_protes_0fail_score = protes_0fail_score[sorted_ids]
    sorted_protes_1fail_score = protes_1fail_score[sorted_ids]
    print('{:} high consistnecy genes:'.format(high_consistency_num))
    for i, gene in enumerate(sorted_protes):
        print(
            protes_consistency_counts[sorted_ids[i]], gene, 
            '{:.3f} {:.3f}'.format(sorted_protes_0fail_score[i], sorted_protes_1fail_score[i])
        )
    data_utils.write_protes(sorted_protes, join(result_path, 'selected_protes.txt'))

def univariate_cox_regression(x, times, events):
    data_dict = {}
    data_dict['x'] = x
    data_dict['event_time'] = times
    data_dict['event'] = events
    df = pd.DataFrame(data=data_dict)
    cph = CoxPHFitter()
    cph.fit(df, duration_col='event_time', event_col='event')
    return cph.params_.tolist()[0], cph.hazard_ratios_.tolist()[0], cph.summary['p'].tolist()[0]

def dataset_cox_regression(dataset, feat_name, time_name, event_name):
    hr_list, p_list = [], []
    for feat_id in progressbar.progressbar(range(len(dataset[feat_name]))):
        coef, hr, p = univariate_cox_regression(dataset['data'][:, feat_id], dataset[time_name], dataset[event_name])
        hr_list.append(hr)
        p_list.append(p)

    # res = Parallel(n_jobs=n_jobs)(delayed(univariate_cox_regression)(
    #         dataset['data'][:, feat_id], dataset[time_name], dataset[event_name]
    #     )for feat_id in progressbar.progressbar(range(len(dataset[feat_name])))
    # )
    return np.asarray(hr_list), np.asarray(p_list)