import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm
import os
import data_utils
import model_utils
from model_utils import train_torch_model, test_torch_model, port_mlp_to_ch, port_ch_to_mlp, proxy_train_mlp, get_CSMIA_case_by_case_results, CSMIA_attack
from data_utils import oneHotCatVars, filter_random_data_by_conf_score
from whitebox_attack import neuron_output, make_neuron_output_data, roc_curve_plot, get_LOMIA_case_1_correct_examples, Top10CorrNeurons, wb_corr_attacks
from disparate_vulnerability_utils import get_accuracy, get_indices_by_conditions, subgroup_vulnerability_distance_vector, subgroup_vulnerability_distance, get_subgroup_disparity, plot_subgroup_disparity, improved_subgroup_attack, get_subgroup_disparity_baseline, get_top_dist_indices, get_disparity_by_subgroup
import shap
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

import matplotlib as mpl

# Setting the font family, size, and weight globally
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 8
mpl.rcParams['font.weight'] = 'light'


class MIAExperiment:
    def __init__(self, *args, **kwargs):
        self.sampling_condition_dict = kwargs.get('sampling_condition_dict', None)
        self.sensitive_column = kwargs.get('sensitive_column', 'MAR')

        for key, value in kwargs.items():
            setattr(self, key, value)

        if not hasattr(self, 'name'):
            self.name = 'Census19'
        self.ds = data_utils.CensusWrapper(
                    filter_prop="none", ratio=float(0.5), split="all", name=self.name, sampling_condition_dict=self.sampling_condition_dict, sensitive_column=self.sensitive_column,
                    additional_meta=None)
        (self.x_tr, self.y_tr), (self.x_te, self.y_te), self.cols = self.ds.load_data()
        self.X_train = pd.DataFrame(self.x_tr, columns=self.cols)
        self.X_test = pd.DataFrame(self.x_te, columns=self.cols)
        self.y_tr_onehot = self.ds.ds.y_enc.transform(self.y_tr).toarray()
        self.y_te_onehot = self.ds.ds.y_enc.transform(self.y_te).toarray()

    def __str__(self):
        return self.ds.ds.filenameroot
    
    def __repr__(self):
        return self.ds.ds.filenameroot
    
    def get_value_count_report(self):
        df = self.ds.ds.original_df
        df = df[df['is_train'] == 1]
        subgroup_values = df[self.subgroup_column].unique().tolist()
        for value in subgroup_values:
            print(f"Subgroup: {value}")
            # print(df[df[self.subgroup_column] == value].columns)
            # print(df[df[self.subgroup_column] == value][[self.sensitive_column, self.y_column]])
            new_df = df[df[self.subgroup_column] == value][[self.sensitive_column, self.y_column]]
            print(new_df.value_counts())
            # print(df[df[self.subgroup_column == value]][[self.sensitive_column, self.y_column]].corr())


    def get_mutual_information_between_sens_and_y(self):
        df = self.ds.ds.original_df
        df = df[df['is_train'] == 1]
        subgroup_values = df[self.subgroup_column].unique().tolist()
        mutual_info_dict = {}
        for value in subgroup_values:
            print(f"Subgroup: {value}")
            # All the features except y column
            X = df[df[self.subgroup_column] == value].drop([self.y_column], axis=1)
            y = df[df[self.subgroup_column] == value][[self.y_column]]
            # print(mutual_info_classif(X, y, discrete_features=True))
            mutual_info_dict[value] = mutual_info_classif(X, y, discrete_features=True)
        return mutual_info_dict


experiments = { f"corr_btn_sens_and_out_{(i, j)}":  MIAExperiment(sampling_condition_dict = 
    {
            'correlation': 0,
            'subgroup_col_name': 'SEX',
            'marginal_prior': 1,
            'corr_btn_sens_and_output_per_subgroup': (i, j),
            # 'fixed_corr_in_test_data': True
    }, shortname = f"Corr_btn_sens_and_output_for_male_({i})_for_female_({j})"
) for (i, j) in [(k, k) for k in [-0.4, -0.35, -0.3, -0.25, -0.2][:]]}
# ) for i in [-0.4, -0.3, -0.2, -0.1, 0][:1] for j in [-0.4, -0.3, -0.2, -0.1, 0][3:4]}


save_model = True

for experiment_key in experiments:
    experiment = experiments[experiment_key]
    print(f"Training classifier for experiment: {experiment}")
    try:
        experiment.clf_only_on_test = model_utils.load_model(f'<PATH_TO_MODEL>/{experiment.ds.ds.filenameroot}_target_model_only_on_test_baseline_comp.pkl')
        print(f"Loaded classifier for experiment from file: {experiment}")
    except:
        # clf = model_utils.get_model(max_iter=500, hidden_layer_sizes=(256, 256))
        base_model = model_utils.get_model(max_iter=40)
        experiment.clf_only_on_test = copy.deepcopy(base_model)
        experiment.clf_only_on_test.fit(experiment.X_test, experiment.y_te_onehot)

        if save_model:
            model_utils.save_model(experiment.clf_only_on_test, f'<PATH_TO_MODEL>/{experiment.ds.ds.filenameroot}_target_model_only_on_test_baseline_comp.pkl')


def imputation_attack(experiment, aux_count=5000):
    original_df_onehot = experiment.ds.ds.df.copy()

    original_df_onehot = original_df_onehot[original_df_onehot["is_train"]==0].drop("is_train", axis=1).reset_index(drop=True)

    X_attack_orig, y_sens_orig = experiment.ds.ds.get_attack_df(original_df_onehot)
    X_attack_orig = X_attack_orig.astype(float)
    y_sens_orig_onehot = experiment.ds.ds.sensitive_enc.transform(y_sens_orig.to_numpy().ravel().reshape(-1, 1)).toarray()

    aux_df_onehot = experiment.ds.ds.df.copy()

    aux_df_onehot = aux_df_onehot[aux_df_onehot["is_train"]==1].drop("is_train", axis=1).reset_index(drop=True)

    X_attack_aux, y_sens_aux = experiment.ds.ds.get_attack_df(aux_df_onehot)
    aux_samp_indices = np.random.choice(X_attack_aux.index, aux_count, False)
    X_attack_aux, y_sens_aux = X_attack_aux.iloc[aux_samp_indices], y_sens_aux[aux_samp_indices]
    X_attack_aux = X_attack_aux.astype(float)
    y_sens_aux_onehot = experiment.ds.ds.sensitive_enc.transform(y_sens_aux.to_numpy().ravel().reshape(-1, 1)).toarray()

    inv_clf = model_utils.get_model(max_iter=40)
    inv_clf.fit(X_attack_aux, y_sens_aux_onehot)

    y_pred = np.argmax(clf.predict(X_attack_orig), axis=1)

    result_dict = get_disparity_by_subgroup(attack_type='INV', ds=experiment.ds, subgroup_columns=['SEX'], X_att_query=X_attack_orig, y_att_query=y_sens_orig, metric='accuracy', clf = inv_clf)
    experiment.inv_clf = inv_clf

    return result_dict


for aux_count in [1000, 5000, 10000, 50000]:
    print(f'\nNumber of samples in aux data: {aux_count}\n')
    for experiment_key in experiments:
        experiment = experiments[experiment_key]
        print(experiment.shortname)
        # print(imputation_attack(experiment, aux_count=aux_count))
        result_dict = imputation_attack(experiment, aux_count=aux_count)
        print(f"ASR: {result_dict['SEX']['original']}")


            
