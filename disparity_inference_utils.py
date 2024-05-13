import numpy as np
from fairlearn.reductions import ExponentiatedGradient
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import seaborn as sns

def get_indices_by_group_condition(X, conditions):
    X_new = X.copy()
    for key, value in conditions.items():
        X_new = X_new.loc[X_new[f"{key}_{value}"] == 1]
    return X_new.index

def get_corr_btn_sens_and_out_per_subgroup(experiment, X, y, conditions):
    sensitive_column = experiment.ds.ds.meta["sensitive_column"]
    indices = get_indices_by_group_condition(X, conditions)
    X_new = X.loc[indices]
    y_new = y[indices]
    X_new = X_new[[f'{sensitive_column}_1']].to_numpy().ravel()
    y_new = y_new.ravel()
    return np.corrcoef(X_new, y_new)[0, 1]

def get_confidence_array(experiment, X, y, model, get_conf_fun=None):
    sensitive_column = experiment.ds.ds.meta["sensitive_column"]
    sensitive_values = experiment.ds.ds.meta["sensitive_values"]
    sensitive_columns_onehot = [f'{sensitive_column}_{i}' for i in range(len(sensitive_values))]
    Xs = [X.copy() for _ in range(len(sensitive_values))]
    for i in range(len(Xs)):
        Xs[i][sensitive_columns_onehot] = 0
        Xs[i][f'{sensitive_column}_{i}'] = 1

    if get_conf_fun is not None:
        y_confs = np.array([get_conf_fun(model, X, y) for X in Xs]).T
    elif isinstance(model, ExponentiatedGradient):
        y_confs = np.array([np.max(predict_proba_for_mitiagtor(model, X), axis=1) for X in Xs]).T
    elif isinstance(model, DecisionTreeClassifier):
        y_confs = np.array([np.max(model.predict_proba(X)[1], axis=1) for X in Xs]).T
    else:
        y_confs = np.array([np.max(model.predict_proba(X), axis=1) for X in Xs]).T

    return y_confs


def draw_confidence_array_scatter(experiment, confidence_array, y, num_points=100, style_fun=None):
    y_values = np.array(experiment.ds.ds.meta["y_values"]).astype(int)
    indices_by_y_values = {y_value: np.where(y.ravel() == y_value)[0] for y_value in y_values}
    colors = ['grey', 'black']
    markers = ['x', 'o']
    scatter_kws_list = [
        dict(s=18),
        dict(s=12, facecolor='none', edgecolor='black')
    ]
    for y_value in [1, 0]:
        sns.regplot(x = confidence_array[indices_by_y_values[y_value], 1][:num_points], y = confidence_array[indices_by_y_values[y_value], 0][:num_points], marker=markers.pop(), color=colors.pop(), line_kws=dict(linewidth=0.5), scatter_kws=scatter_kws_list.pop())

    plt.xlabel('Confidence Score when queried with sensitive attribute = 1')
    plt.ylabel('Conf. Score when queried with sensitive attribute = 0')
    if style_fun is not None:
        style_fun(ax)
    plt.show()


def get_slopes(experiment, confidence_array, y):
    y_values = np.array(experiment.ds.ds.meta["y_values"]).astype(int)
    indices_by_y_values = {y_value: np.where(y.ravel() == y_value)[0] for y_value in y_values}

    return [np.polyfit(confidence_array[indices_by_y_values[y_value], 1], confidence_array[indices_by_y_values[y_value], 0], 1)[0] for y_value in [1, 0]]


def get_angular_difference(experiment, confidence_array, y):
    slopes = get_slopes(experiment, confidence_array, y)
    return np.arctan(np.abs(slopes[1] - slopes[0]) / (1 + slopes[1] * slopes[0])) * np.sign(slopes[1] - slopes[0])


def calculate_stds(experiment, confidence_array, y):
    y_values = np.array(experiment.ds.ds.meta["y_values"]).astype(int)
    indices_by_y_values = {y_value: np.where(y.ravel() == y_value)[0] for y_value in y_values}

    return np.array([np.std(confidence_array[indices_by_y_values[y_value]], axis=0) for y_value in y_values])


    