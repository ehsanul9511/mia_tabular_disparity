from tqdm import tqdm
import os
import data_utils
import model_utils
from attack_utils import get_CSMIA_case_by_case_results, CSMIA_attack, LOMIA_attack
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
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate
import pickle
# import utils
import copy

import matplotlib as mpl

# Setting the font family, size, and weight globally
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 8
mpl.rcParams['font.weight'] = 'light'


def bcorr_training(experiment, X_train, y_tr):
    original_test_df = experiment.ds.ds.original_df[experiment.ds.ds.original_df['is_train']==0].copy().reset_index(drop=True).drop(['is_train'], axis=1)

    p = -0.1
    x = original_test_df[original_test_df['SEX']==0][['MAR', 'PINCP']].value_counts().to_numpy().min()
    n = (x * 4) // (1 + p)

    temp_indices = experiment.ds.ds.sample_data_matching_correlation(original_test_df, p1=-0.1, p2=-0.1, n=2*n, subgroup_col_name='SEX', transformed_already=True, return_indices_only=True)

    experiment.X_test_balanced_corr = experiment.X_test.loc[temp_indices].reset_index(drop=True)
    experiment.y_te_balanced_corr = experiment.y_te[temp_indices]
    experiment.y_te_onehot_balanced_corr = experiment.y_te_onehot[temp_indices]