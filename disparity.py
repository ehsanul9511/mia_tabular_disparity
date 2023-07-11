from tqdm import tqdm
import os
import data_utils
import model_utils
from model_utils import train_torch_model, test_torch_model, port_mlp_to_ch, port_ch_to_mlp, proxy_train_mlp
from data_utils import oneHotCatVars
from whitebox_attack import neuron_output, make_neuron_output_data, roc_curve_plot, get_LOMIA_case_1_correct_examples, Top10CorrNeurons, wb_corr_attacks
from disparate_vulnerability_utils import get_accuracy, get_indices_by_conditions, subgroup_vulnerability_distance_vector, subgroup_vulnerability_distance, get_subgroup_disparity, plot_subgroup_disparity, improved_subgroup_attack, get_subgroup_disparity_baseline, get_top_dist_indices, get_disparity_by_subgroup
import shap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import tabulate
import pickle
# import utils


subgroup_col_name = 'RAC1P'
split_ratio_first_subgroup = 0.5

sampling_condition_dict_list = [
    {
        'condition': lambda x: x[subgroup_col_name] < 2,
        'sample_size': round(split_ratio_first_subgroup * 100000),
    },
    {
        'condition': lambda x: x[subgroup_col_name] == 3,
        'sample_size': round(split_ratio_first_subgroup * 100000),
    },
]

# ds = data_utils.CensusWrapper(
#             filter_prop="none", ratio=float(0.5), split="all", name="Census19")

ds = data_utils.CensusWrapper(
            filter_prop="none", ratio=float(0.5), split="all", name="Census19", sampling_condition_dict_list=sampling_condition_dict_list, sensitive_column='DEAR')
(x_tr, y_tr), (x_te, y_te), cols = ds.load_data()
X_train = pd.DataFrame(x_tr, columns=cols)
y_tr_onehot = ds.ds.y_enc.transform(y_tr).toarray()


# ds.ds.filenameroot = ds.ds.name
ds.ds.filenameroot = ds.ds.name + f"_{subgroup_col_name}_{round(100*split_ratio_first_subgroup)}_{round(100*(1-split_ratio_first_subgroup))}_minority_categorized"

save_model = True

try:
    clf = model_utils.load_model(f'models/{ds.ds.filenameroot}_target_model.pkl')
except:
    # clf = model_utils.get_model(max_iter=500, hidden_layer_sizes=(256, 256))
    clf = model_utils.get_model(max_iter=500)
    clf.fit(X_train, y_tr_onehot)

    if save_model:
        model_utils.save_model(clf, f'models/{ds.ds.filenameroot}_target_model.pkl')


attack_types = ['LOMIA', 'INV', 'PCA', 'GRAD', 'NEUR_OURS', 'NEUR_IMP']

subgroup_columns = ['RAC1P', 'SEX']


def LOMIA(all_sensitive_columns = ['MAR', 'DEAR', 'DEYE', 'DREM', 'DPHY'], metric='auc'):

    subgroup_disparity_dicts = {}

    for sensitive_column in all_sensitive_columns:
        temp_ds = data_utils.CensusWrapper(
                filter_prop="none", ratio=float(0.5), split="all", name="Census19", sampling_condition_dict_list=sampling_condition_dict_list, sensitive_column=sensitive_column)
        temp_ds.ds.filenameroot = temp_ds.ds.name + f"_{subgroup_col_name}_{round(100*split_ratio_first_subgroup)}_{round(100*(1-split_ratio_first_subgroup))}_minority_categorized"
        X, y = temp_ds.ds.get_attack_df()

        X = X.astype(float)

        # subgroup_columns = ['RAC1P']

        attack_clf = model_utils.load_model(f'models/{temp_ds.ds.filenameroot}_LOMIA_attack_{temp_ds.ds.meta["sensitive_column"]}_model.pkl')

        subgroup_disparity_dict = get_disparity_by_subgroup(ds=ds, X_att_query=X, y_att_query=y, subgroup_columns=subgroup_columns, clf=attack_clf, metric=metric)

        subgroup_disparity_dicts[sensitive_column] = subgroup_disparity_dict

    return subgroup_disparity_dicts


def INV(all_sensitive_columns = ['MAR', 'DEAR', 'DEYE', 'DREM', 'DPHY'], metric='auc'):
    save_model = True

    subgroup_disparity_dicts = {}

    for sensitive_column in all_sensitive_columns:
        temp_ds = data_utils.CensusWrapper(
                filter_prop="none", ratio=float(0.5), split="all", name="Census19", sampling_condition_dict_list=sampling_condition_dict_list, sensitive_column=sensitive_column)
        temp_ds.ds.filenameroot = temp_ds.ds.name + f"_{subgroup_col_name}_{round(100*split_ratio_first_subgroup)}_{round(100*(1-split_ratio_first_subgroup))}_minority_categorized"
        X, y = temp_ds.ds.get_attack_df()

        X = X.astype(float)

        y_onehot = temp_ds.ds.sensitive_enc.transform(y.to_numpy().ravel().reshape(-1, 1)).toarray()

        # subgroup_columns = ['RAC1P']

        try:
            inv_clf = model_utils.load_model(f'models/{temp_ds.ds.filenameroot}_inverse_model_{sensitive_column}.pkl')

        except:
            x_tr, x_te, y_tr, y_te = train_test_split(X, y_onehot, test_size=0.9, random_state=42)

            inv_clf = model_utils.get_model(max_iter=500)
            inv_clf.fit(x_tr, y_tr)
            # inv_clf = model_utils.proxy_train_mlp(x_tr.to_numpy(), y_tr, epochs=100)
            # inv_clfs[test_size] = inv_clf

            acc = 100 * inv_clf.score(x_te, y_te)
            # print(f'Inverse accuracy with test size : {acc}')

            if save_model:
                model_utils.save_model(inv_clf, f'models/{temp_ds.ds.filenameroot}_inverse_model_{sensitive_column}.pkl')
        
        subgroup_disparity_dict = get_disparity_by_subgroup(attack_type='INV', ds=ds, subgroup_columns=subgroup_columns, X_att_query=X, y_att_query=y, metric=metric, clf=inv_clf)
        subgroup_disparity_dicts[sensitive_column] = subgroup_disparity_dict


    return subgroup_disparity_dicts

def GRAD(all_sensitive_columns = ['default', 'DEAR', 'DEYE', 'DREM', 'DPHY'], metric='auc'):

    subgroup_disparity_dicts = {}

    for sens_col in all_sensitive_columns:
        temp_ds = data_utils.CensusWrapper(
                filter_prop="none", ratio=float(0.5), split="all", name="Census19", sampling_condition_dict_list=sampling_condition_dict_list, sensitive_column=sens_col)
        temp_ds.ds.filenameroot = temp_ds.ds.name + f"_{subgroup_col_name}_{round(100*split_ratio_first_subgroup)}_{round(100*(1-split_ratio_first_subgroup))}_minority_categorized"
        X, y = temp_ds.ds.get_attack_df()

        orig_grads = np.load(f'<PATH_TO_GRADS>/{ds.ds.filenameroot}_{sens_col}_orig_grads.npy')
        grads_after_flip = np.load(f'<PATH_TO_GRADS>/{ds.ds.filenameroot}_{sens_col}_grads_after_flip.npy')
        sens_attrs = np.load(f'<PATH_TO_GRADS>/{ds.ds.filenameroot}_{sens_col}_sens_attrs.npy')
        scores = np.load(f'<PATH_TO_GRADS>/{ds.ds.filenameroot}_{sens_col}_scores.npy')

        tpr, fpr, thresholds = roc_curve(sens_attrs, scores)
        # get the threshold with tpr = 0.5
        best_threshold = thresholds[np.argmin(np.abs(tpr - 0.5))]
        # best_threshold = thresholds[np.argmax(tpr - fpr)]


        # subgroup_columns = ['RAC1P', 'SEX']

        subgroup_disparity_dict = get_disparity_by_subgroup(attack_type='GRAD', ds=ds, X_att_query=X, y_att_query=y, y_pred=scores, threshold=best_threshold, subgroup_columns=subgroup_columns, metric=metric)

        subgroup_disparity_dicts[sens_col] = subgroup_disparity_dict

    return subgroup_disparity_dicts


def PCA_attack(all_sensitive_columns = ['default', 'DEAR', 'DEYE', 'DREM', 'DPHY'], metric='auc'):
    save_model = True

    subgroup_disparity_dicts = {}

    for sensitive_column in all_sensitive_columns:
        temp_ds = data_utils.CensusWrapper(
                filter_prop="none", ratio=float(0.5), split="all", name="Census19", sampling_condition_dict_list=sampling_condition_dict_list, sensitive_column=sensitive_column)
        temp_ds.ds.filenameroot = temp_ds.ds.name + f"_{subgroup_col_name}_{round(100*split_ratio_first_subgroup)}_{round(100*(1-split_ratio_first_subgroup))}_minority_categorized"
        X, y = temp_ds.ds.get_attack_df()

        X = X.astype(float)
        y_onehot = ds.ds.sensitive_enc.transform(y.to_numpy().ravel().reshape(-1, 1)).toarray()

        # subgroup_columns = ['RAC1P']

        try:
            pca_clf = model_utils.load_model(f'models/{temp_ds.ds.filenameroot}_pca_model_{sensitive_column}.pkl')
            with open(f'models/{temp_ds.ds.filenameroot}_pca_model_pca_{sensitive_column}.pkl', 'rb') as f:
                pca = pickle.load(f)
        except:
            x_tr, x_te, y_tr, y_te = train_test_split(X, y_onehot, test_size=0.9, random_state=42)

            pca = PCA(n_components=2)
            pca.fit(x_tr)
                
            x_tr, x_te = map(lambda x: pd.DataFrame(pca.transform(x), index=x.index), [x_tr, x_te])

            pca_clf = model_utils.get_model(max_iter=500)
            pca_clf.fit(x_tr, y_tr)

            if save_model:
                model_utils.save_model(pca_clf, f'models/{temp_ds.ds.filenameroot}_pca_model_{sensitive_column}.pkl')
                with open(f'models/{temp_ds.ds.filenameroot}_pca_model_pca_{sensitive_column}.pkl', 'wb') as f:
                    pickle.dump(pca, f)

        X = pd.DataFrame(pca.transform(X), index=X.index)
        
        subgroup_disparity_dict = get_disparity_by_subgroup(attack_type='PCA', ds=ds, subgroup_columns=subgroup_columns, X_att_query=X, y_att_query=y, metric=metric, clf=pca_clf)
        subgroup_disparity_dicts[sensitive_column] = subgroup_disparity_dict

    return subgroup_disparity_dicts

def NEUR(all_sensitive_columns = ['MAR', 'DEAR', 'DEYE', 'DREM', 'DPHY'], metric='auc', attack_subtype='NEUR_OURS', target_clf=None):
    save_model = True

    subgroup_disparity_dicts = {}

    for sensitive_column in all_sensitive_columns:
        temp_ds = data_utils.CensusWrapper(
                filter_prop="none", ratio=float(0.5), split="all", name="Census19", sampling_condition_dict_list=sampling_condition_dict_list, sensitive_column=sensitive_column)
        temp_ds.ds.filenameroot = temp_ds.ds.name + f"_{subgroup_col_name}_{round(100*split_ratio_first_subgroup)}_{round(100*(1-split_ratio_first_subgroup))}_minority_categorized"
        X, y = temp_ds.ds.get_attack_df()

        X = X.astype(float)

        y_onehot = temp_ds.ds.sensitive_enc.transform(y.to_numpy().ravel().reshape(-1, 1)).toarray()

        # subgroup_columns = ['RAC1P']


        x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.9, random_state=42)

        x_n_tr = make_neuron_output_data(temp_ds, x_tr, clf, temp_ds.ds.y_columns)
        x_n_te = make_neuron_output_data(temp_ds, x_te, clf, temp_ds.ds.y_columns)

        if attack_subtype == 'NEUR_OURS':
            try:
                neur_clf = model_utils.load_model(f'models/{temp_ds.ds.filenameroot}_neurours_model_{sensitive_column}.pkl')
            except:
                neur_clf = model_utils.get_model(max_iter=500)
                neur_clf.fit(x_n_tr, y_tr)
                # inv_clf = model_utils.proxy_train_mlp(x_tr.to_numpy(), y_tr, epochs=100)
                # inv_clfs[test_size] = inv_clf

                acc = 100 * neur_clf.score(x_n_te, y_te)
                # print(f'NeurOurs accuracy with test size : {acc}')

                if save_model:
                    model_utils.save_model(neur_clf, f'models/{temp_ds.ds.filenameroot}_neurours_model_{sensitive_column}.pkl')
        else:
            neur_clf = wb_corr_attacks(x_n_tr, y_tr)
        
        
        subgroup_disparity_dict = get_disparity_by_subgroup(attack_type=attack_subtype, ds=temp_ds, subgroup_columns=subgroup_columns, X_att_query=X, y_att_query=y, metric=metric, clf=neur_clf, df=X, MLP=target_clf)
        subgroup_disparity_dicts[sensitive_column] = subgroup_disparity_dict

    return subgroup_disparity_dicts
    

subgroup_disparity_dicts_by_attack_type = {}

for metric in ['auc', 'recall', 'precision', 'accuracy']:
    subgroup_disparity_dicts_by_attack_type[metric] = {}
    for attack_type in attack_types:
        if attack_type == 'INV':
            subgroup_disparity_dicts = INV(metric=metric)
        elif attack_type == 'GRAD':
            subgroup_disparity_dicts = GRAD(metric=metric)
        elif attack_type == 'PCA':
            subgroup_disparity_dicts = PCA_attack(metric=metric)
        elif attack_type == 'LOMIA':
            subgroup_disparity_dicts = LOMIA(metric=metric)
        elif attack_type in ['NEUR_OURS', 'NEUR_IMP']:
            subgroup_disparity_dicts = NEUR(attack_subtype=attack_type, target_clf=clf, metric=metric)

        print(f'Attack type : {attack_type}')
        print(subgroup_disparity_dicts)
    

        subgroup_disparity_dicts_by_attack_type[metric][attack_type] = subgroup_disparity_dicts

