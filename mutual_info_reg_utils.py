import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import numpy as np
import model_utils

class EnsembleMLPClassifier(MLPClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, Xs, ys):
        # Train an MLPClassifier on each split
        self.classifiers = []
        for i in tqdm(range(len(Xs))):
            X_split = Xs[i]
            y_split = ys[i]
            
            # Create a new MLPClassifier for each split
            clf = model_utils.get_model()  # Use current instance parameters
            clf.fit(X_split, y_split)
            
            # Store the trained classifier
            self.classifiers.append(clf)
    
    def predict_proba(self, X):
        # Collect probability predictions from each classifier
        probas = np.zeros((X.shape[0], 2))  # Assuming binary classification (2 classes)
        
        for clf in self.classifiers:
            probas += clf.predict_proba(X)
        
        # Average the probability predictions
        probas /= len(self.classifiers)
        return probas
    
    def predict(self, X, one_hot=True):
        # Use the averaged probability predictions to predict the class labels
        probas = self.predict_proba(X)
        # Get the predicted class for each sample
        predictions = np.argmax(probas, axis=1)
        
        if one_hot:
            # Convert the predicted class labels to one-hot encoding
            n_classes = probas.shape[1]  # Number of classes
            one_hot_encoded = np.zeros((predictions.size, n_classes))
            one_hot_encoded[np.arange(predictions.size), predictions] = 1
            return one_hot_encoded
        else:
            # Return the predicted class labels
            return predictions
    
    def score(self, X, y):
        # Predict the class labels and calculate accuracy
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


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

    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
    def predict_proba(self, x: torch.Tensor):
        return self.forward(x)
        
class MLPClassifierMutualInfoReg(nn.Module):
    def __init__(self, n_in_features=37, n_feat_dim=10, n_out_features=2):
        super(MLPClassifierMutualInfoReg, self).__init__()
        layers = [
            nn.Linear(in_features=n_in_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=n_feat_dim),
            # nn.Softmax(dim=1)
        ]
        self.layers = nn.Sequential(*layers)
        self.k = n_feat_dim//2
        self.st_layer = nn.Linear(in_features=n_feat_dim, out_features=self.k*2)
        self.classifier = nn.Linear(in_features=self.k, out_features=n_out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        
        statis = self.st_layer(x)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        std = torch.functional.F.softplus(std-5)
        # torch.manual_seed(42)
        # seed = hash_tensor(x)
        # torch.manual_seed(seed)
        eps = torch.FloatTensor(std.size()).normal_().to(x.device)
        x = mu + eps * std
        x = self.classifier(x)
        x = self.softmax(x)
        return x, mu, std
    
    def predict_proba(self, x: torch.Tensor):
        return self.forward(x)[0]

    def train_mir(self, train_loader, beta=0.1, selective_reg=False, epochs=10):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in tqdm(range(epochs)):
            for batch_idx, (data, target) in enumerate(train_loader):
                self.train()
                optimizer.zero_grad()
                data, target = data.to('mps'), target.to('mps')
                output, mu, std = self(data[:, :-1])
                info_loss = - 0.5 * (1 + 2 * (std+1e-7).log() - mu.pow(2) - std.pow(2)).sum(dim=1)
                if selective_reg:
                    info_loss = info_loss * data[:, -1]
                info_loss = info_loss.mean()
                loss = nn.BCELoss()(output, target) + beta * info_loss
                loss.backward()
                optimizer.step()

    # test on test set
    def test_mir(self, X_test, y_te_onehot):
        x_te = X_test.values
        dataset = torch.utils.data.TensorDataset(torch.tensor(x_te).float(), torch.tensor(y_te_onehot).float())
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=x_te.shape[0], shuffle=False)

        self.eval()
        y_pred = []
        y_true = []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to('mps'), target.to('mps')
            output, _, _ = self(data)
            y_pred.append(output.cpu().detach().numpy())
            y_true.append(target.cpu().detach().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)

        return accuracy_score(y_true, y_pred)
        return(classification_report(y_true, y_pred))


def add_vuln_attrib(X_train, all_vuln_scores_rounded):
    X_train_w_vuln = X_train.copy()
    X_train_w_vuln['vuln'] = pd.Series(all_vuln_scores_rounded, index=X_train_w_vuln.index)
    return X_train_w_vuln


def get_trainloader_with_vuln(X_train, y_tr_onehot, all_vuln_scores_rounded, batch_size=128):
    X_train_w_vuln = add_vuln_attrib(X_train, all_vuln_scores_rounded)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_w_vuln.values).float(), torch.tensor(y_tr_onehot).float())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def get_trainloader_with_nonvuln(X_train, y_tr_onehot, all_vuln_scores_rounded, batch_size=128):
    X_train_w_vuln = add_vuln_attrib(X_train, all_vuln_scores_rounded)
    X_train_w_vuln = X_train_w_vuln[X_train_w_vuln['vuln']==0]
    # print(y_tr_onehot[X_train_w_vuln.index, :].shape)
    y_tr_onehot = y_tr_onehot[X_train_w_vuln.index, :]
    # print(X_train_w_vuln.index)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_w_vuln.values).float(), torch.tensor(y_tr_onehot).float())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def split_datasets(X_train, y_tr_onehot, all_vuln_scores_rounded, n_splits=5):
    # Step 1: Add the vulnerability attribute to X_train
    X_train_w_vuln = add_vuln_attrib(X_train, all_vuln_scores_rounded)

    # Step 2: Filter records with vuln == 0 and vuln == 1
    X_train_w_vuln_0 = X_train_w_vuln[X_train_w_vuln['vuln'] == 0]
    X_train_w_vuln_1 = X_train_w_vuln[X_train_w_vuln['vuln'] == 1]

    # Step 3: Balance the dataset
    if len(X_train_w_vuln_1) < len(X_train_w_vuln_0):
        X_train_w_vuln_1 = X_train_w_vuln_1.sample(len(X_train_w_vuln_0), replace=True, random_state=42)
    else:
        X_train_w_vuln_0 = X_train_w_vuln_0.sample(len(X_train_w_vuln_1), replace=True, random_state=42)

    # Step 4: Combine and shuffle the balanced datasets without resetting the index
    X_train_w_vuln_balanced = pd.concat([X_train_w_vuln_0, X_train_w_vuln_1]).sample(frac=1, random_state=42)

    X_train_w_vuln_balanced = X_train_w_vuln_balanced.drop(['vuln'], axis=1)

    # Step 5: Match y_tr_onehot with the indices of the shuffled X_train_w_vuln_balanced
    y_tr_onehot_balanced = y_tr_onehot[X_train_w_vuln_balanced.index]

    # Step 6: Split X_train_w_vuln_balanced and y_tr_onehot_balanced into n small dataframes
    # Compute the number of batches based on the batch size
    batch_size = int(np.ceil(len(X_train_w_vuln_balanced) / n_splits))

    # Split both X_train and y_tr simultaneously to ensure matching indices
    X_train_split = [X_train_w_vuln_balanced.iloc[i * batch_size: (i + 1) * batch_size] for i in range(n_splits)]
    y_tr_split = [y_tr_onehot_balanced[i * batch_size: (i + 1) * batch_size] for i in range(n_splits)]

    return X_train_split, y_tr_split