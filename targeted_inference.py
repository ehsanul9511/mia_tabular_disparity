from tqdm import tqdm
import os
import data_utils
import model_utils
from attack_utils import get_CSMIA_case_by_case_results, CSMIA_attack, LOMIA_attack, get_LOMIA_results, imputation_attack
from data_utils import oneHotCatVars, filter_random_data_by_conf_score
from vulnerability_score_utils import get_vulnerability_score, draw_hist_plot
from experiment_utils import MIAExperiment
from disparity_inference_utils import get_confidence_array, draw_confidence_array_scatter, get_indices_by_group_condition, get_corr_btn_sens_and_out_per_subgroup, get_slopes, get_angular_difference, calculate_stds, get_mutual_info_btn_sens_and_out_per_subgroup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate
import pickle
# import utils
import copy
from scipy.stats import kendalltau, spearmanr

def get_subgroup_vals_sorted_by_risk_imputation(experiment, aux_df, subgroup_col_name = 'occupation', is_train=1, metric='accuracy'):
    imputation_pred_test = imputation_attack(experiment, aux_df_onehot=aux_df, subgroup_column='occupation', pred_on_train=False, is_train=is_train)
    
    aux_ground_truth = aux_df[[f'{experiment.sensitive_column}_{experiment.sensitive_positive}']].to_numpy().ravel()
    correlation_vs_ang_diff = {}
    subgroup_vals = experiment.ds.ds.original_df[subgroup_col_name].unique()
    for i in subgroup_vals:
        condition = {subgroup_col_name: i}
        fcondition = f'{condition}'
        correlation_vs_ang_diff[fcondition] = {}
        indices = get_indices_by_group_condition(aux_df, condition)
        correlation_vs_ang_diff[fcondition]['subgroup_val'] = i
        correlation_vs_ang_diff[fcondition]['indices_count'] = len(indices)
        # correlation_vs_ang_diff[fcondition]['imputation_performance'] = accuracy_score(aux_ground_truth[indices], imputation_pred_test[indices])
        correlation_vs_ang_diff[fcondition]['imputation_performance'] = experiment.score(aux_ground_truth[indices], imputation_pred_test[indices], metric=metric)

    correlation_vs_ang_diff_df = pd.DataFrame.from_dict(correlation_vs_ang_diff, orient='index')

    # print(correlation_vs_ang_diff_df)
    occupation_vals_sorted_imputation = correlation_vs_ang_diff_df.sort_values(by='imputation_performance', ascending=False)[['subgroup_val']].to_numpy().ravel()

    return occupation_vals_sorted_imputation

def single_attribute_based_targeted_imputation(experiment, aux_df, subgroup_col_name = 'occupation', is_train=1, kappas=[1, 0.5, 0.375, 0.25, 0.1, 0.05], metric='accuracy'):
    occupation_vals_sorted_imputation = get_subgroup_vals_sorted_by_risk_imputation(experiment, aux_df, subgroup_col_name=subgroup_col_name, metric=metric).tolist()

    imputation_pred = imputation_attack(experiment, aux_df_onehot=aux_df, subgroup_column='occupation', is_train=is_train)

    targeted_attack_by_state_dict = {}
    # sensitive_column_index = list(experiment.X_train.columns).index(f'{experiment.sensitive_column}_1')
    conditions = [{subgroup_col_name: occupation_vals_sorted_imputation[:i+1]} for i in range(len(occupation_vals_sorted_imputation))]
    cumul_frac_of_total_records = [len(get_indices_by_group_condition(experiment.X_train, condition))/experiment.X_train.shape[0] for condition in conditions]
    # return cumul_frac_of_total_records
    for i in kappas:
        j = np.argmin(np.abs(np.array(cumul_frac_of_total_records)-i))
        condition = {subgroup_col_name: occupation_vals_sorted_imputation[:j+1]}
        fcondition = f'{condition}'
        targeted_attack_by_state_dict[fcondition] = {}
        indices = get_indices_by_group_condition(experiment.X_train, condition)
        frac_of_total_records = len(indices)/experiment.X_train.shape[0]
        targeted_attack_by_state_dict[frac_of_total_records] = {}
        # targeted_attack_by_state_dict[frac_of_total_records]['imputation_attack_accuracy'] = (experiment.sens_val_ground_truth_imputation[indices] == imputation_pred[indices]).sum()/len(indices)
        targeted_attack_by_state_dict[frac_of_total_records]['imputation_attack_accuracy'] = experiment.score(experiment.sens_val_ground_truth_imputation[indices], imputation_pred[indices], metric=metric)

    return pd.DataFrame.from_dict(targeted_attack_by_state_dict, orient='index')

def nested_attribute_based_targeted_imputation(experiment, aux_df, subgroup_cols, is_train=1, kappas=[0.5, 0.25, 0.125, 0.06125, 0.0306125], metric='accuracy'):
    imputation_pred = imputation_attack(experiment, aux_df_onehot=aux_df, subgroup_column='occupation', is_train=is_train)
    correlation_vs_ang_diff = {}
    subgroup_vals_sorted_dict = {col: get_subgroup_vals_sorted_by_risk_imputation(experiment, aux_df, col).tolist() for col in subgroup_cols}
    # print(subgroup_vals_sorted_dict)
    nested_conditions = [({}, {})]
    for i in range(len(subgroup_cols)):
        subgroup_col = subgroup_cols[i]
        subgroup_vals_sorted = subgroup_vals_sorted_dict[subgroup_col]
        kappa = kappas[i]
        conditions = [{subgroup_col: subgroup_vals_sorted[:j+1]} for j in range(len(subgroup_vals_sorted))]
        conditions = [condition | nested_conditions[-1][0] for condition in conditions]
        # print(conditions)
        cumul_frac_of_total_records = [len(get_indices_by_group_condition(experiment.X_train, condition))/experiment.X_train.shape[0] for condition in conditions]
        # print(cumul_frac_of_total_records)
        # print(np.where(np.array(cumul_frac_of_total_records) > kappa))
        # j = np.argmin(np.abs(np.array(cumul_frac_of_total_records)-kappa))
        j = np.where(np.array(cumul_frac_of_total_records) > kappa)[0][0] if np.any(np.array(cumul_frac_of_total_records) > kappa) else 0
        nested_conditions.append((conditions[j], conditions[j-1]))

    for i, (condition, prev_condition) in enumerate(nested_conditions):
        fcondition = f'{condition}'
        correlation_vs_ang_diff[fcondition] = {}
        indices = get_indices_by_group_condition(experiment.X_train, condition)
        frac_of_total_records = len(indices)/experiment.X_train.shape[0]
        if i>0 and frac_of_total_records > kappas[i-1]:
            indices = get_indices_by_group_condition(experiment.X_train, prev_condition)
            frac_of_total_records = len(indices)/experiment.X_train.shape[0]
        #     print(prev_condition)
        # else:
        #     print(condition)
        correlation_vs_ang_diff[frac_of_total_records] = {}
        correlation_vs_ang_diff[frac_of_total_records]['i'] = i
        # correlation_vs_ang_diff[frac_of_total_records]['attack_accuracy'] = (experiment.sens_val_ground_truth_imputation[indices] == imputation_pred[indices]).sum()/len(indices)
        correlation_vs_ang_diff[frac_of_total_records]['attack_accuracy'] = experiment.score(experiment.sens_val_ground_truth_imputation[indices], imputation_pred[indices], metric=metric)

    return pd.DataFrame.from_dict(correlation_vs_ang_diff, orient='index')

def get_angular_difference_for_each_subgroup_val(experiment, subgroup_col_name):
    correlation_vs_ang_diff = {}
    subgroup_vals = experiment.ds.ds.original_df[subgroup_col_name].unique()

    for i in subgroup_vals:
        condition = {subgroup_col_name: i}
        fcondition = f'{condition}'
        indices = get_indices_by_group_condition(experiment.X_case_2, condition)
        try:
            angular_difference = get_angular_difference(experiment, experiment.confidence_array_case_2[indices], experiment.y_case_2[indices])
            correlation_vs_ang_diff[fcondition] = {}
            correlation_vs_ang_diff[fcondition]['subgroup_val'] = i
            correlation_vs_ang_diff[fcondition]['angular_difference'] = angular_difference
        except:
            continue

    correlation_vs_ang_diff_df = pd.DataFrame.from_dict(correlation_vs_ang_diff, orient='index')
    
    if experiment.ds.ds.name in ['Census19', 'Adult']:
        return correlation_vs_ang_diff_df.sort_values(by='angular_difference', ascending=True)
    else:
        return correlation_vs_ang_diff_df.sort_values(by='angular_difference', ascending=False)

def get_angular_difference_range_for_subgroup(experiment, subgroup_col_name):
    angular_differences = get_angular_difference_for_each_subgroup_val(experiment, subgroup_col_name)[['angular_difference']].to_numpy()
    return np.max(angular_differences) - np.min(angular_differences)

def get_subgroup_vals_sorted_by_risk(experiment, subgroup_col_name):
    return get_angular_difference_for_each_subgroup_val(experiment, subgroup_col_name)[['subgroup_val']].to_numpy().ravel()

def single_attribute_based_targeted_ai(experiment, sens_pred, subgroup_col_name = 'occupation', kappas=[1, 0.5, 0.375, 0.25, 0.1, 0.05], metric='accuracy'):
    occupation_vals_sorted = get_subgroup_vals_sorted_by_risk(experiment, subgroup_col_name=subgroup_col_name).tolist()
    # print(occupation_vals_sorted)

    targeted_attack_by_state_dict = {}
    # sensitive_column_index = list(experiment.X_train.columns).index(f'{experiment.sensitive_column}_1')
    conditions = [{subgroup_col_name: occupation_vals_sorted[:i+1]} for i in range(len(occupation_vals_sorted))]
    cumul_frac_of_total_records = [len(get_indices_by_group_condition(experiment.X_train, condition))/experiment.X_train.shape[0] for condition in conditions]
    # return cumul_frac_of_total_records
    for i in kappas:
        j = np.argmin(np.abs(np.array(cumul_frac_of_total_records)-i))
        condition = {subgroup_col_name: occupation_vals_sorted[:j+1]}
        fcondition = f'{condition}'
        targeted_attack_by_state_dict[fcondition] = {}
        indices = get_indices_by_group_condition(experiment.X_train, condition)
        frac_of_total_records = len(indices)/experiment.X_train.shape[0]
        targeted_attack_by_state_dict[frac_of_total_records] = {}
        # targeted_attack_by_state_dict[frac_of_total_records]['attack_accuracy'] = (experiment.sens_val_ground_truth[indices] == sens_pred[indices]).sum()/len(indices)
        targeted_attack_by_state_dict[frac_of_total_records]['attack_accuracy'] = experiment.score(experiment.sens_val_ground_truth[indices], sens_pred[indices], metric=metric)

    print(condition)

    return pd.DataFrame.from_dict(targeted_attack_by_state_dict, orient='index')

def nested_attribute_based_targeted_ai(experiment, sens_pred, subgroup_cols, kappas=[0.5, 0.25, 0.125, 0.06125, 0.0306125], metric='accuracy'):
    correlation_vs_ang_diff = {}
    subgroup_vals_sorted_dict = {col: get_subgroup_vals_sorted_by_risk(experiment, col).tolist() for col in subgroup_cols}
    # print(subgroup_vals_sorted_dict)
    nested_conditions = [({}, {})]
    for i in range(len(subgroup_cols)):
        subgroup_col = subgroup_cols[i]
        subgroup_vals_sorted = subgroup_vals_sorted_dict[subgroup_col]
        kappa = kappas[i]
        conditions = [{subgroup_col: subgroup_vals_sorted[:j+1]} for j in range(len(subgroup_vals_sorted))]
        conditions = [condition | nested_conditions[-1][0] for condition in conditions]
        # print(conditions)
        cumul_frac_of_total_records = [len(get_indices_by_group_condition(experiment.X_train, condition))/experiment.X_train.shape[0] for condition in conditions]
        # print(cumul_frac_of_total_records)
        # print(np.where(np.array(cumul_frac_of_total_records) > kappa))
        # j = np.argmin(np.abs(np.array(cumul_frac_of_total_records)-kappa))
        j = np.where(np.array(cumul_frac_of_total_records) > kappa)[0][0] if np.any(np.array(cumul_frac_of_total_records) > kappa) else 0
        jp = j-1 if j>0 else 0
        nested_conditions.append((conditions[j], conditions[jp]))

    # print(nested_conditions)
    for i, (condition, prev_condition) in enumerate(nested_conditions):
        # print(condition)
        # print(prev_condition)
        fcondition = f'{condition}'
        # correlation_vs_ang_diff[fcondition] = {}
        indices = get_indices_by_group_condition(experiment.X_train, condition)
        frac_of_total_records = len(indices)/experiment.X_train.shape[0]
        # print(frac_of_total_records)
        if i>0 and frac_of_total_records > kappas[i-1]:
            condition = prev_condition
            indices = get_indices_by_group_condition(experiment.X_train, condition)
            frac_of_total_records = len(indices)/experiment.X_train.shape[0]
            # print(frac_of_total_records)
        #     print(prev_condition)
        # else:
        # print(condition.keys())
        correlation_vs_ang_diff[frac_of_total_records] = {}
        correlation_vs_ang_diff[frac_of_total_records]['i'] = i
        # correlation_vs_ang_diff[frac_of_total_records]['attack_accuracy'] = (experiment.sens_val_ground_truth[indices] == sens_pred[indices]).sum()/len(indices)
        correlation_vs_ang_diff[frac_of_total_records]['attack_accuracy'] = experiment.score(experiment.sens_val_ground_truth[indices], sens_pred[indices], metric=metric)

    return pd.DataFrame.from_dict(correlation_vs_ang_diff, orient='index')

    