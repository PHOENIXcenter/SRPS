import os, copy, time, argparse, platform, progressbar, pickle
import numpy as np
import pandas as pd
import viz_utils as viz_utils
import matplotlib.pyplot as plt
import tensorflow as tf
from os.path import join
from shutil import copyfile
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import models.SRPS as SRPS
import models.SourceOnly as DNN
import data_utils as data_utils
from sklearn.manifold import TSNE
from survival_correlated_data import generate_survival_correlated_synthetic_data

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--data', default='HCC')
parser.add_argument('--experiment', default='Jiang2Gao')
parser.add_argument('--fig_name', default='benchmarking_real')
parser.add_argument('--fold_num', type=int, default=5)
parser.add_argument('--model_name', default='SRPS_hdim-10_bl_lr-1e-04_loss_su-4e-02_regu-1e-04_dropout_rl-1e-01_dropout-8e-01_no_early_stop-False')
parser.add_argument('--subtype_num', type=int, default=3)
parser.add_argument('--data_path', default='data')
args = parser.parse_args()

def benchmarking_real():
    save_path = join(args.data_path, args.experiment, 'formal_results', 'benchmarking')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    experiment_path = join(args.data_path, args.experiment)
    method_param_df = pd.read_csv(
        join(experiment_path, 'param_table.csv'), 
        header=[0], 
        delimiter = ','
    )
    methods = method_param_df['method'].to_list()
    result_list = method_param_df['param'].to_list()
    ids = [methods.index(method) for method in [
        'RandomForest', 'SourceOnly', 'RandomForestHarmony', 'DANN', 'deepCLife', 'SRPS'
    ]]
    result_list = [result_list[idx] for idx in ids]
    method_list = ['RF', 'DNN', 'RFH', 'DANN', 'semi-\ndeepCLife', 'SRPS']

    if args.data == 'HCC':
        cohort_name_list = ['Jiang', 'Gao'] 
    elif args.data == 'LUAD':
        cohort_name_list = ['Xu', 'Gillette']
    elif args.data == 'HCC_LUAD':
        cohort_name_list = ['Jiang', 'Xu']
    else:
        print('invalid data name!')
        exit()

    col_list = [
        cohort_name_list[0] + '_Accuracy', 
        cohort_name_list[1] + '_ssGSEA_Similarity', 
        cohort_name_list[1] + '_OS_Log-rank score', 
        cohort_name_list[1] + '_DFS_Log-rank score', 
    ]

    metric_list = [
        'Accuracy\n(source)', 
        'Similarity\n(target)', 
        'Log-rank\nscore (OS)', 'Log-rank\nscore (RFS)',
    ]

    score_df_list = [pd.read_csv(
        join(args.data_path, args.experiment, 'seed' + str(seed), 'compare_table.csv')
    ) for seed in range(args.fold_num)]

    row_ids = [score_df_list[0]['method'].to_list().index(result) for result in result_list]

    # heatmap
    score_arr_list = [df[col_list].loc[row_ids, :].to_numpy() for df in score_df_list]
    score_arrs = np.stack(score_arr_list, axis=-1) # methods * metrics * seeds
    metrics= [' '.join(col.split('_')[1:]) for col in col_list]
    viz_utils.heatmap_vectors_with_stats(
        data_arr=score_arrs, 
        x_labels=metric_list, 
        y_labels=method_list,
        rotation=0., 
        fig_size=(4, 4), 
        save_path=join(save_path, 'heatmap.png')
    )

    # best km and ssgsea plots
    w, h = 4, 3
    methods_best_seed = np.argmax(score_arrs[:, metric_list.index('Log-rank\nscore (OS)'), :], axis=-1)
    print(score_arrs.shape, methods_best_seed.shape)
    for i, result in enumerate(result_list):
        best_seed = methods_best_seed[i]
        best_result_path = join(args.data_path, args.experiment, 'seed'+str(best_seed), result)
        fold_num = len([x for x in os.listdir(best_result_path) if 'fold-' in x])
        df = pd.concat([pd.read_csv(join(best_result_path, 'fold-' + str(fold) + '.csv')) for fold in range(fold_num)])

        result_save_path = join(save_path, result)
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)
        print(result_save_path)
        cohorts = df['cohort'].to_numpy().astype(int)
        subtype_num = np.amax(df['label'].to_numpy()).astype(int) + 1
        for cohort_id in range(np.amax(cohorts) + 1):
            if cohort_id > 0:
                # plot km curves
                assignments = df['assignment'].to_numpy()[cohorts == cohort_id]
                for time_name, event_name in [['OS', 'status'], ['DFS', 'recurrence']]:
                    times = df[time_name].to_numpy()[cohorts == cohort_id]
                    events = df[event_name].to_numpy()[cohorts == cohort_id]

                    fig, ax = plt.subplots(figsize=(w, h))
                    title = cohort_name_list[cohort_id] + '_' + time_name 
                    viz_utils.plot_km_curve_custom(
                        times, 
                        events, 
                        assignments, 
                        subtype_num,
                        ax, 
                        title='',
                        text_bias=24,
                        clip_time=60
                    )
                    fig.tight_layout()
                    plt.savefig(join(result_save_path, 'km_' + title + '.png'))
                    plt.close()
            # copy ssgsea plots
            try:
                file_name = 'GSEA_' + cohort_name_list[cohort_id] + '_heatmap.png'
                copyfile(join(best_result_path, file_name), join(result_save_path, file_name))
            except Exception as e:
                pass
            else:
                print('copy error')


def benchmarking_synthetic():
    save_path = join(args.data_path, args.experiment, 'formal_results', 'benchmarking')
    if not os.path.exists(save_path):
        os.makedirs(save_path) 

    method_list = ['RF', 'DNN', 'RFH', 'DANN', 'semi-deepCLife', 'SRPS(soft)', 'SRPS(no baseline)', 'SRPS']
    col_list = [
        'Domain0_Accuracy', 'Domain1_Accuracy',
        'Domain0_OS_Log-rank score', 'Domain1_OS_Log-rank score'
    ]
    metric_list = [
        'Accuracy (source)', 'Accuracy (target)', 
        'Log-rank score (source)', 'Log-rank score (target)'
    ]
    lv_method_metric_seed = []
    for level in [1, 3]:
        experiment = join(args.data_path, args.experiment, 'batch_effect_lv' + str(level))
        method_param_df = pd.read_csv(
            join(experiment, 'param_table.csv'), 
            header=[0], 
            delimiter = ','
        )
        methods = method_param_df['method'].to_list()
        result_list = method_param_df['param'].to_list()
        ids = [methods.index(method) for method in [
            'RandomForest', 'SourceOnly', 'RandomForestHarmony', 'DANN', 'deepCLife', 'SRPS(soft)', 'SRPS(no', 'SRPS', 
        ]]
        result_list = [result_list[idx] for idx in ids]
        score_df_list = [pd.read_csv(
            join(experiment, 'seed' + str(seed), 'compare_table.csv')
        ) for seed in range(args.fold_num)]

        row_ids = [score_df_list[0]['method'].to_list().index(result) for result in result_list]
        score_arr_list = [df[col_list].loc[row_ids, :].to_numpy() for df in score_df_list]
        method_metric_seed = np.stack(score_arr_list, axis=-1) # methods * metrics * seeds 
        lv_method_metric_seed.append(method_metric_seed)
    lv_method_metric_seed = np.stack(lv_method_metric_seed, axis=0)
    viz_utils.batch_effect_bar_plots(
        lv_method_metric_seed[:, :, [0, 1], :], 
        methods=method_list, 
        metric='Accuracy', 
        lvs=['without batch effect', 'with batch effect'],
        fig_size=(8, 3), 
        ylimit_list =[[0.4, 1.0], [0.4, 1.0]],
        save_path=save_path
    )

    viz_utils.batch_effect_bar_plots(
        lv_method_metric_seed[:, :, [2, 3], :], 
        methods=method_list, 
        metric='Log-rank score', 
        lvs=['without batch effect', 'with batch effect'],
        fig_size=(8, 3), 
        ylimit_list =[None, None],
        save_path=save_path
    )

def toy_test():
    save_path = join(args.data_path, args.experiment)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seed_num = len([x for x in os.listdir(save_path) if 'seed' in x])

    size = 11
    batch_noise_rate_step = 0.4
    os_swap_rate_step = 0.02
    acc_os = np.zeros((size, size))
    for os_swap_i in progressbar.progressbar(range(size)):
        for batch_noise_j in range(size):
            datasets = generate_survival_correlated_synthetic_data(
                seed=0, 
                n_sample=200,
                censor_rate=0., 
                batch_noise_rate=0 * batch_noise_rate_step, 
                os_swap_rate=os_swap_i * os_swap_rate_step, 
                viz=False, 
                save_path=None
            )

            # re-arrange subtypes to make OS perfectly stratified by subtype
            subtypes = datasets[1]['subtypes']
            assignments = copy.deepcopy(subtypes)
            OS = datasets[1]['OS']
            status = datasets[1]['status']
            assignments[OS < 30.] = 1
            assignments[OS >= 30.] = 0
            acc = np.mean(assignments == subtypes)
            acc_os[os_swap_i, batch_noise_j] = acc

    os_swap_rate = np.arange(size) * os_swap_rate_step
    batch_noise_rate = np.arange(size) * batch_noise_rate_step

    acc_SRPS_list, acc_DNN_list = [], []
    for seed in range(seed_num):
        acc_SRPS = np.loadtxt(join(save_path, 'seed'+str(seed), 'target_domain_acc_SRPS.csv'), delimiter=',')[:, 2]
        acc_SRPS = np.reshape(acc_SRPS, (size, size))
        acc_SRPS_list.append(acc_SRPS)

        acc_DNN = np.loadtxt(join(save_path, 'seed'+str(seed), 'target_domain_acc_DNN.csv'), delimiter=',')[:, -1]
        acc_DNN = np.reshape(acc_DNN, (size, size))
        acc_DNN_list.append(acc_DNN)

    acc_SRPS = np.mean(np.stack(acc_SRPS_list, axis=0), axis=0)
    acc_DNN = np.mean(np.stack(acc_DNN_list, axis=0), axis=0)

    # acc_os = np.tile(np.expand_dims(acc_os, axis=1), (1, size))
    cmap = cm.coolwarm

    vmax = 1.0
    vmin = 0.7
    _ = viz_utils.imshow(acc_SRPS, vmax=vmax, vmin=vmin, size=(3.5, 3), title='Acc(SRPS)', save_path=join(save_path, 'acc_SRPS.png'))
    _ = viz_utils.imshow(acc_DNN, vmax=vmax, vmin=vmin, size=(3.5, 3), title='Acc(DNN)', save_path=join(save_path, 'acc_DNN.png'))
    _ = viz_utils.imshow(acc_os, vmax=vmax, vmin=vmin, size=(3.5, 3), title='Acc(Survival)', save_path=join(save_path, 'acc_os.png'))

    vmax = 0.2
    vmin = -0.2
    ax = viz_utils.imshow(
        acc_SRPS - acc_DNN,  
        # vmax=np.amax(acc_SRPS - acc_DNN),
        # vmin=-np.amax(acc_SRPS - acc_DNN),
        vmax=vmax,
        vmin=vmin,
        size=(3.5, 3),
        title='Acc(SRPS) - Acc(DNN)', 
        save_path=join(save_path, 'acc_SRPS-DNN.png')
    )
    _ = viz_utils.imshow(
        acc_os - acc_SRPS, 
        # vmax=np.amax(acc_SRPS - acc_os),
        # vmin=-np.amax(acc_SRPS - acc_os),
        vmax=vmax,
        vmin=vmin,
        size=(3.5, 3), 
        title='Acc(Survival) - Acc(SRPS)', 
        save_path=join(save_path, 'acc_OS-SRPS.png')
    )

    ax = viz_utils.imshow(
        acc_os - acc_DNN,  
        # vmax=np.amax(acc_os - acc_DNN),
        # vmin=-np.amax(acc_os - acc_DNN),
        vmax=vmax,
        vmin=vmin,
        size=(3.5, 3),
        title='Acc(Survival) - Acc(DNN)', 
        save_path=join(save_path, 'acc_OS-DNN.png')
    )

    # calculating the significance
    for data1, data2 in [[acc_SRPS, acc_DNN], [acc_os, acc_SRPS], [acc_os, acc_DNN]]:
        os_swap_i = size
        batch_noise_j = 0
        selected_acc_1 = np.reshape(data1[:os_swap_i, batch_noise_j:], (-1))
        selected_acc_2 = np.reshape(data2[:os_swap_i, batch_noise_j:], (-1))
        # _, _, p_all = data_utils.welch_ttest(selected_acc_1, selected_acc_2, 'greater')
        _, _, p_all = data_utils.one_sample_ttest(selected_acc_1 - selected_acc_2, 0, alternative='greater')

        os_swap_i = 6
        batch_noise_j = 5
        selected_acc_1 = np.reshape(data1[:os_swap_i, batch_noise_j:], (-1))
        selected_acc_2 = np.reshape(data2[:os_swap_i, batch_noise_j:], (-1))
        # _, _, p_upper_right = data_utils.welch_ttest(selected_acc_1, selected_acc_2, 'greater')
        _, _, p_upper_right = data_utils.one_sample_ttest(selected_acc_1 - selected_acc_2, 0, alternative='greater')

        print('p_all={:.3f}, p_upper_right={:.3e}'.format(p_all, p_upper_right))

def model_explaination():
    SRPS_model = SRPS.SRPSNet([(3, False, None, 1e-4)])
    DNN_model = DNN.SourceOnly({'encoder_layer_num': 0, 'encoder_hdim': 0, 'class_layer_num': 0, 'class_hdim':0, 
        'subtype_num': 3, 'use_bias': False, 'regularization': 0.
    })

    platform_sys = platform.system()
    if platform_sys == 'Linux':
        file_name = 'all_datasets_linux.p'
    else:
        file_name = 'all_datasets_windows.p'

    pickle_file_path = join(args.data_path, file_name)
    data_path = args.data_path
    save_path = join(data_path, 'model_explaination')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    datasets_dict, gene2prote_dict, prote2gene_dict = pickle.load(open(pickle_file_path, 'rb'))
    datasets = datasets_dict['HCC-Jiang2Gao']
    data = datasets[0]['data']
    data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-5)
    _ = SRPS_model.predict(data)
    _ = DNN_model.predict(data)
    genes = [prote2gene_dict[prote] for prote in datasets[0]['protes']]

    model_name = args.model_name
    DNN_model_name = 'SourceOnly_encoder_layer_num-0_encoder_hdim-20_class_layer_num-0_class_hdim-10_activation-None_learning_rate-1e-02_dropout-8e-01'

    # the model weights of each subtype
    weight_results = viz_utils.subtype_model_weight_viz(
        SRPS_model, 
        experiment_path=join(save_path, 'models'),
        save_path=save_path,
        model_name=model_name,
        genes=genes,
        seed_num=5,
        fold_num=5
    )

    # feature protes of each subtype
    DEP_results = viz_utils.DEP_viz(
        experiment_path=join(save_path, 'models'),
        save_path=save_path,
        model_name=model_name,
        datasets=datasets,
        genes=genes,
        seed_num=5,
        fold_num=5
    )

    for s in range(3, 4):
        for weight_key in weight_results.keys():
            if 'S'+str(s) in weight_key:
                for DEP_key in DEP_results.keys():
                    if 'S'+str(s) in DEP_key and 'logFC' in DEP_key:
                        viz_utils.viz_correlation(
                            weight_results[weight_key],
                            DEP_results[DEP_key],
                            # weight_key,
                            '\u0394weight',
                            # DEP_key,
                            'Log2 (fold change)',
                            None,
                            join(save_path, 'corr_'+weight_key+'_'+DEP_key+'.png'),
                            top_prote=True
                        )

    # the prognosis discrimination comparison between top 30 proteins of different models
    viz_utils.compare_correlation_between_delta_weight_and_hr(
        [SRPS_model, DNN_model],
        [join(save_path, 'models'), join(save_path, 'models')],
        [model_name, DNN_model_name],
        datasets,
        save_path
    )

    # investigate the PPIC protein
    model_name = args.model_name
    for gene in ['PPIC']:
        prote = gene2prote_dict[gene]
        viz_utils.prote_viz(
            datasets=datasets, 
            seed_num=5, 
            fold_num=5, 
            experiment_path=join(save_path, 'models'),
            model_name=model_name,
            gene=gene,
            prote=prote,
            save_path=save_path
        )

if __name__ == '__main__':
    
    if args.fig_name == 'benchmarking_real':
        benchmarking_real()
    elif args.fig_name == 'benchmarking_synthetic':
        benchmarking_synthetic()
    elif args.fig_name == 'toy_test':
        toy_test()
    elif args.fig_name == 'model_explaination':
        model_explaination()
    else:
        print('unknown figure name')

