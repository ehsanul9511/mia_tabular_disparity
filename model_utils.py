import numpy as np
from tqdm import tqdm
from typing import List
import torch as ch
import torch.nn as nn
import os
# from utils import check_if_inside_cluster, make_affinity_features
from joblib import load, dump
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.metrics import roc_auc_score
import platform

device = ch.device("cuda" if ch.cuda.is_available() else "mps" if platform.machine() == "arm64" else "cpu")


BASE_MODELS_DIR = "<PATH_TO_MODELS>"
ACTIVATION_DIMS = [32, 16, 8, 2]


class PortedMLPClassifier(nn.Module):
    def __init__(self, n_in_features=37, n_out_features=2):
        super(PortedMLPClassifier, self).__init__()
        layers = [
            nn.Linear(in_features=n_in_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=n_out_features),
            nn.Softmax(dim=1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: ch.Tensor,
                latent: int = None,
                get_all: bool = False,
                detach_before_return: bool = True,
                on_cpu: bool = False):
        """
        Args:
            x: Input tensor of shape (batch_size, 42)
            latent: If not None, return only the latent representation. Else, get requested latent layer's output
            get_all: If True, return all activations
            detach_before_return: If True, detach the latent representation before returning it
            on_cpu: If True, return the latent representation on CPU
        """
        if latent is None and not get_all:
            return self.layers(x)

        if latent not in [0, 1, 2] and not get_all:
            raise ValueError("Invald interal layer requested")

        if latent is not None:
            # First three hidden layers correspond to outputs of
            # Model layers 1, 3, 5
            latent = (latent * 2) + 1
        valid_for_all = [1, 3, 5, 6]

        latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Append activations for all layers (post-activation only)
            if get_all and i in valid_for_all:
                if detach_before_return:
                    if on_cpu:
                        latents.append(x.detach().cpu())
                    else:
                        latents.append(x.detach())
                else:
                    if on_cpu:
                        latents.append(x.cpu())
                    else:
                        latents.append(x)
            if i == latent:
                if on_cpu:
                    return x.cpu()
                else:
                    return x

        return latents

    
# def train_torch_model(model=None, X=None, y=None, epochs=100, lr=0.01):
#     """
#         Train PyTorch model on given data
#     """
#     if model is None:
#         model = PortedMLPClassifier(n_in_features=X.shape[1], n_out_features=y.shape[1])
#     model = model.to(device)
#     if X is None or y is None:
#         return model
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = ch.optim.Adam(model.parameters(), lr=lr)
#     X = ch.tensor(X, dtype=ch.float32).to(device)
#     y = ch.tensor(y, dtype=ch.long).to(device)
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         y_pred = model(X)
#         loss = loss_fn(y_pred, y)
#         loss.backward()
#         optimizer.step()
#     return model


def port_mlp_to_ch(clf):
    """
        Extract weights from MLPClassifier and port
        to PyTorch model.
    """
    nn_model = PortedMLPClassifier(n_in_features=clf.coefs_[0].shape[0],
                                   n_out_features=clf.coefs_[-1].shape[1])
    i = 0
    for (w, b) in zip(clf.coefs_, clf.intercepts_):
        w = ch.from_numpy(w.T).float()
        b = ch.from_numpy(b).float()
        nn_model.layers[i].weight = nn.Parameter(w)
        nn_model.layers[i].bias = nn.Parameter(b)
        i += 2  # Account for ReLU as well

    # nn_model = nn_model.cuda()
    nn_model = nn_model.to(device)
    return nn_model


def port_ch_to_mlp(nn_model, clf=None):
    """
        Extract weights from PyTorch model and port
        to MLPClassifier.
    """
    # if clf is None:
    #     clf = get_model()
    #     y_shape_1 = nn_model.layers[-1].weight.shape[0]
    #     dtype = nn_model.layers[-1].weight.dtype
    #     hidden_layer_sizes = [nn_model.layers[i].weight.shape[1] for i in range(0, len(nn_model.layers), 2)]
    #     layer_units = [nn_model.layers[0].weight.shape[0]] + hidden_layer_sizes + [y_shape_1]
    #     clf.set_params(hidden_layer_sizes=layer_units, activation='relu', solver='adam', alpha=0.0001)
    #     clf._initialize(np.zeros(1,y_shape_1), layer_units, dtype)

    for i, layer in enumerate(nn_model.layers):
        if i % 2 == 0:
            clf.coefs_[i // 2] = layer.weight.detach().cpu().numpy().T
            clf.intercepts_[i // 2] = layer.bias.detach().cpu().numpy()

    return clf

def proxy_train_mlp(X, y, epochs=100, lr=0.01, l1_reg=0.0):
    """
        Train PyTorch model on given data
    """
    nn_model = train_torch_model(model=None, X=X, y=y, epochs=epochs, lr=lr, l1_reg=l1_reg)
    clf = get_model(max_iter=1)
    clf.fit(X, y)
    clf = port_ch_to_mlp(nn_model, clf)
    return clf


def convert_to_torch(clfs):
    """
        Port given list of MLPClassifier models to
        PyTorch models
    """
    return np.array([port_mlp_to_ch(clf) for clf in clfs], dtype=object)


# def layer_output(data, MLP, layer=0, get_all=False):
#     """
#         For a given model and some data, get output for each layer's activations < layer.
#         If get_all is True, return all activations unconditionally.
#     """
#     L = data.copy()
#     all = []
#     for i in range(layer):
#         L = ACTIVATIONS['relu'](
#             np.matmul(L, MLP.coefs_[i]) + MLP.intercepts_[i])
#         if get_all:
#             all.append(L)
#     if get_all:
#         return all
#     return L


def layer_output(data, MLP, layer=0, get_all=False):
    """
        For a given model and some data, get output for each layer's activations < layer.
        If get_all is True, return all activations unconditionally.
    """
    X = ch.tensor(data, dtype=ch.float64)
    L = X
    all = []
    for i in range(layer):
        L = ch.relu(ch.matmul(L, ch.tensor(MLP.coefs_[i])) + ch.tensor(MLP.intercepts_[i]))
        if get_all:
            all.append(L)
    if get_all:
        return [L.detach().numpy() for L in all]
    return L.detach().numpy()


# Load models from directory, return feature representations
def get_model_representations(folder_path, label, first_n=np.inf,
                              n_models=1000, start_n=0,
                              fetch_models: bool = False,
                              shuffle: bool = True,
                              models_provided: bool = False):
    """
        If models_provided is True, folder_path will actually be a list of models.
    """
    if models_provided:
        models_in_folder = folder_path
    else:
        models_in_folder = os.listdir(folder_path)

    if shuffle:
        # Shuffle
        np.random.shuffle(models_in_folder)

    # Pick only N models
    models_in_folder = models_in_folder[:n_models]

    w, labels, clfs = [], [], []
    for path in tqdm(models_in_folder):
        if models_provided:
            clf = path
        else:
            clf = load_model(os.path.join(folder_path, path))
        if fetch_models:
            clfs.append(clf)

        # Extract model parameters
        weights = [ch.from_numpy(x) for x in clf.coefs_]
        dims = [w.shape[0] for w in weights]
        biases = [ch.from_numpy(x) for x in clf.intercepts_]
        processed = [ch.cat((w, ch.unsqueeze(b, 0)), 0).float().T
                     for (w, b) in zip(weights, biases)]

        # Use parameters only from first N layers
        # and starting from start_n
        if first_n != np.inf:
            processed = processed[start_n:first_n]
            dims = dims[start_n:first_n]

        w.append(processed)
        labels.append(label)

    labels = np.array(labels)

    w = np.array(w, dtype=object)
    labels = ch.from_numpy(labels)

    if fetch_models:
        return w, labels, dims, clfs
    return w, labels, dims


def get_model(max_iter=40,
              hidden_layer_sizes=(32, 16, 8),
              random_state=42):
    """
        Create new MLPClassifier model
    """
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter,
                        random_state=random_state)
    return clf


def get_models(folder_path, n_models=1000, shuffle=True):
    """
        Load models from given directory.
    """
    paths = os.listdir(folder_path)
    if shuffle:
        paths = np.random.permutation(paths)
    paths = paths[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = load_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def save_model(clf, path):
    dump(clf, path)


def load_model(path):
    return load(path)


def get_models_path(property, split, value=None):
    if value is None:
        return os.path.join(BASE_MODELS_DIR, property, split)
    return os.path.join(BASE_MODELS_DIR,  property, split, value)


def get_model_activation_representations(
        models: List[PortedMLPClassifier],
        data, label, detach: bool = True,
        verbose: bool = True):
    w = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator)
    for model in iterator:
        activations = model(data, get_all=True,
                            detach_before_return=detach)
        # Skip last feature (logit)
        activations = activations[:-1]

        w.append([act.float() for act in activations])
    labels = np.array([label] * len(w))
    labels = ch.from_numpy(labels)

    # Make numpy object (to support sequence-based indexing)
    w = np.array(w, dtype=object)

    # Get dimensions of feature representations
    dims = [x.shape[1] for x in w[0]]

    return w, labels, dims


def make_activation_data(models_pos, models_neg, seed_data,
                         detach=True, verbose=True, use_logit=False):
    # Construct affinity graphs
    pos_model_scores = make_affinity_features(
        models_pos, seed_data,
        detach=detach, verbose=verbose,
        use_logit=use_logit)
    neg_model_scores = make_affinity_features(
        models_neg, seed_data,
        detach=detach, verbose=verbose,
        use_logit=use_logit)
    # Convert all this data to loaders
    X = ch.cat((pos_model_scores, neg_model_scores), 0)
    Y = ch.cat((ch.ones(len(pos_model_scores)),
                ch.zeros(len(neg_model_scores))))
    return X, Y


def make_affinity_feature(model, data, use_logit=False, detach=True, verbose=True):
    """
         Construct affinity matrix per layer based on affinity scores
         for a given model. Model them in a way that does not
         require graph-based models.
    """
    # Build affinity graph for given model and data
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # Start with getting layer-wise model features
    model_features = model(data, get_all=True, detach_before_return=detach)
    layerwise_features = []
    for i, feature in enumerate(model_features):
        # Old (before 2/4)
        # Skip logits if asked not to use (default)
        # if not use_logit and i == (len(model_features) - 1):
            # break
        scores = []
        # Pair-wise iteration of all data
        for i in range(len(data)-1):
            others = feature[i+1:]
            scores += cos(ch.unsqueeze(feature[i], 0), others)
        layerwise_features.append(ch.stack(scores, 0))

    # New (2/4)
    # If asked to use logits, convert them to probability scores
    # And then consider them as-it-is (instead of pair-wise comparison)
    if use_logit:
        logits = model_features[-1]
        probs = ch.sigmoid(logits)
        layerwise_features.append(probs)

    concatenated_features = ch.stack(layerwise_features, 0)
    return concatenated_features


def make_affinity_features(models, data, use_logit=False, detach=True, verbose=True):
    all_features = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Building affinity matrix")
    for model in iterator:
        all_features.append(
            make_affinity_feature(
                model, data, use_logit=use_logit, detach=detach, verbose=verbose)
        )
    return ch.stack(all_features, 0)

def train_torch_model(model=None, X=None, y=None, epochs=100, lr=0.01, l1_reg=0.0):
    """
        Train PyTorch model on given data
    """
    if model is None:
        model = PortedMLPClassifier(n_in_features=X.shape[1], n_out_features=y.shape[1])
    model = model.to(device)
    if X is None or y is None:
        return model
    def l1_loss(model):
        loss = 0.0
        for param in model.parameters():
            loss += ch.sum(ch.abs(param))
        loss = ch.mean(loss)
        return loss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = ch.optim.Adam(model.parameters(), lr=lr)
    X = ch.tensor(X, dtype=ch.float32).to(device)
    y = ch.tensor(np.argmax(y, axis=1), dtype=ch.long).to(device) # Convert multi-target tensor to class labels
    # create dataset and dataloader
    dataset = ch.utils.data.TensorDataset(X, y)
    dataloader = ch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target) + l1_reg * l1_loss(model)
            loss.backward()
            optimizer.step()
    # for epoch in range(epochs):
    #     optimizer.zero_grad()
    #     y_pred = model(X)
    #     loss = loss_fn(y_pred, y) + l1_reg * l1_loss(model)
    #     loss.backward()
    #     optimizer.step()
    return model

def test_torch_model(model, X, y, metric='accuracy'):
    """
        Test PyTorch model on given data
    """
    # device = model_utils.device
    model = model.to(device)
    X = ch.tensor(X, dtype=ch.float32).to(device)
    y = ch.tensor(np.argmax(y, axis=1), dtype=ch.long).to(device) # Convert multi-target tensor to class labels
    y_pred = model(X)
    print(y_pred)
    # test_loss = nn.CrossEntropyLoss()(y_pred, y).item()
    if metric == 'accuracy':
        test_acc = (y_pred.argmax(1) == y).type(ch.float32).mean().item()
    elif metric == 'auc':
        test_acc = roc_auc_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    # return test_loss, test_acc
    return test_acc

def LOMIA_attack(model, X_test, y_test, meta):
    attack_dataset = []
    for i in tqdm(range(len(X_test))):
        # Get the predicted label and true label for this record
        #pred_label = predicted_labels[i]
        # true_label = y_test.iloc[i]
        # true_label = y_enc.transform([y_test.iloc[i]])[0]
        true_label = y_test[i]
        
        # Check if the predicted label matches the true label for only one possible value of the sensitive attribute
        num_matches = 0
        matched_value = None
        # sensitive_values = ["Married", "Single"]
        sensitive_values = meta["sensitive_values"]
        sensitive_attr = meta["sensitive_column"]
        predictions = []
        for sensitive_value in sensitive_values:
            record = X_test.iloc[i:i+1].copy()
            # record[sensitive_attr + "_" + sensitive_value] = 1
            record[f'{sensitive_attr}_{sensitive_value}'] = 1

            for other_value in sensitive_values:
                if other_value != sensitive_value:
                    # record[sensitive_attr + "_" + other_value] = 0
                    record[f'{sensitive_attr}_{other_value}'] = 0
            
            # Check if the predicted label matches the true label for this sensitive value
            # if clf.predict([record])[0] == true_label:
            # prediction = np.argmax(model.predict(record.to_numpy().reshape(1, -1))[0])
            prediction = np.argmax(model.predict(record))
            # print(prediction)
            # if model.predict(record.to_numpy().reshape(1, -1))[0] == true_label:
            if prediction == true_label:
                num_matches += 1
                matched_value = sensitive_value
                
        # If there is only one match, label the record with the matched value
        if num_matches == 1:
            record = X_test.iloc[i:i+1].copy()
            # record[sensitive_attr + "_" + matched_value] = 1
            record[f'{sensitive_attr}_{matched_value}'] = 1

            for other_value in sensitive_values:
                if other_value != matched_value:
                    # record[sensitive_attr + "_" + other_value] = 0
                    record[f'{sensitive_attr}_{other_value}'] = 0
            
            # record[data_dict['y_column']] = (true_label == data_dict['y_pos'])
            record[meta['y_column']] = true_label
            attack_dataset.append(record)
            
    return attack_dataset