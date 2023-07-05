# import utils
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
import requests
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict

BASE_DATA_DIR = "<PATH_TO_DATASET>"

SUPPORTED_PROPERTIES = ["sex", "race", "none"]
PROPERTY_FOCUS = {"sex": "Female", "race": "White"}
SUPPORTED_RATIOS = ["0.0", "0.1", "0.2", "0.3",
                    "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        return key


# US Income dataset
class CensusIncome:
    def __init__(self, name="Adult", path=BASE_DATA_DIR, sensitive_column="default", preload=True, sampling_condition_dict_list=None):
        # self.urls = [
        #     "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        #     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
        #     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        # ]
        # self.columns = [
        #     "age", "workClass", "fnlwgt", "education", "education-num",
        #     "marital-status", "occupation", "relationship",
        #     "race", "sex", "capital-gain", "capital-loss",
        #     "hours-per-week", "native-country", "income"
        # ]
        # self.dropped_cols = ["education", "native-country"]
        self.path = path
        self.name = name
        if name == "Adult":
            self.meta = {
                "train_data_path": os.path.join(self.path, "Adult_35222.csv"),
                "test_data_path": os.path.join(self.path, "Adult_10000.csv"),
                # "all_columns": [
                #     "age", "workClass", "fnlwgt", "education", "education-num",
                #     "marital-status", "occupation", "relationship",
                #     "race", "sex", "capital-gain", "capital-loss",
                #     "hours-per-week", "native-country", "income"
                # ],
                "all_columns": ['work', 'fnlwgt', 'education', 'marital', 'occupation', 'sex',
                                'capitalgain', 'capitalloss', 'hoursperweek', 'race', 'income'
                                ],   
                "cat_columns": ["work", "education", "marital", "occupation", "race", "sex"],
                # "num_columns": ["age", "fnlwgt", "capitalgain", "capitalloss", "hoursperweek"],
                "num_columns": {
                    "zscaler": ["fnlwgt", "capitalgain", "capitalloss"],
                    "minmaxscaler": ["hoursperweek"]
                },
                "y_column": "income",
                "y_positive": ">50K",
                "y_values": ['<=50K', '>50K'],
                "sensitive_column": "marital",
                "sensitive_values": ["Married", "Single"],
                "sensitive_positive": "Married"
            }
        elif name == "Census19":
            if sensitive_column == "default":
                sensitive_column = "MAR"
            self.meta = {
                # "train_data_path": os.path.join(self.path, "Adult_35222.csv"),
                # "test_data_path": os.path.join(self.path, "Adult_10000.csv"),
                "data_path": os.path.join(self.path, "census19.csv"),
                # "all_columns": [
                #     "age", "workClass", "fnlwgt", "education", "education-num",
                #     "marital-status", "occupation", "relationship",
                #     "race", "sex", "capital-gain", "capital-loss",
                #     "hours-per-week", "native-country", "income"
                # ],
                "all_columns": ['AGEP', 'COW', 'SCHL', 'MAR', 'RAC1P', 'SEX', 'DREM', 'DPHY', 'DEAR',
                                'DEYE', 'WKHP', 'WAOB', 'ST', 'PINCP'],   
                "cat_columns": ['COW', 'SCHL', 'MAR', 'RAC1P', 'SEX', 'DREM', 'DPHY', 'DEAR',
                                'DEYE', 'WAOB', 'ST'],
                # "num_columns": ["age", "fnlwgt", "capitalgain", "capitalloss", "hoursperweek"],
                "num_columns": {
                    "zscaler": [],
                    "minmaxscaler": ['AGEP', 'WKHP']
                },
                "y_column": "PINCP",
                "y_positive": '1',
                "y_values": ['0', '1'],
                "sensitive_column": sensitive_column,
                "sensitive_values": [0, 1],
                "sensitive_positive": defaultdict(lambda: 1, {'MAR': 0})[sensitive_column],
                "transformations": [
                    lambda df: df.drop(columns=['PUMA']),
                    lambda df: df.assign(MAR = df['MAR'].apply(lambda x: 1 if x >= 1 else 0)),
                    lambda df: df.assign(RAC1P = df['RAC1P'].apply(lambda x: 0 if x <= 1 else 1)),
                    # lambda df: df.assign(sensitive_column = df[sensitive_column].astype('str')),
                    lambda df: df.assign(PINCP = df['PINCP'].apply(lambda x: '1' if x > 90000 else '0')),
                    # lambda df: df.assign(PINCP = df['PINCP'].astype('str')),
                ]
            }
        elif name == "GSS":
            self.meta = {
                "train_data_path": os.path.join(self.path, "GSS_15235.csv"),
                "test_data_path": os.path.join(self.path, "GSS_5079.csv"),
                "all_columns": ['year', 'marital', 'divorce', 'sex', 'race', 'relig', 'xmovie', 'pornlaw',
                                'childs', 'age', 'educ', 'hapmar'],
                "cat_columns": ['year', 'marital', 'divorce', 'sex', 'race', 'relig', 'xmovie', 'pornlaw'],
                "num_columns": {
                    "zscaler": [],
                    "minmaxscaler": ['childs', 'age', 'educ']
                },
                "y_column": "hapmar",
                "y_positive": "nottoohappy",
                "y_values": ['nottoohappy', 'prettyhappy', 'veryhappy'],
                "sensitive_column": "xmovie",
                "sensitive_values": ["x_yes", "x_no"],
                "sensitive_positive": "x_yes"
            }
        elif name == "NLSY":
            self.meta = {
                "train_data_path": os.path.join(self.path, "nlsy_5096.csv"),
                "test_data_path": os.path.join(self.path, "nlsy_5096.csv"),
                "all_columns": [],
                "cat_columns": ['marital8', 'gender', 'race', 'arrestsdli8', 'drug_marijuana',
                                'smoking8', 'drinking8', 'sexdrugsdli8', 'sexstrng8'],
                "num_columns": {
                    "zscaler": ['incarceration', 'income8'],
                    "minmaxscaler": ['age','children']
                },
                "y_column": "ratelife8",
                "y_positive": "excellent",
                # "y_values": ['excellent', 'verygood', 'good', 'fair', 'poor'],
                "sensitive_column": "drug_marijuana",
                "sensitive_values": ["drug_marijuana_yes", "drug_marijuana_no"],
                "sensitive_positive": "drug_marijuana_yes"
            }

        self.y_columns = [self.meta['y_column'] + "_" + val for val in self.meta['y_values']]

        self.y_mapping_dict = {value: index for index, value in enumerate(self.meta["y_values"])}
                
        # self.download_dataset()
        # self.load_data(test_ratio=0.4)
        if preload:
            self.load_data(test_ratio=0.5, sampling_condition_dict_list=sampling_condition_dict_list)


    # Download dataset, if not present
    def download_dataset(self):
        if not os.path.exists(self.path):
            print("==> Downloading US Census Income dataset")
            os.mkdir(self.path)
            print("==> Please modify test file to remove stray dot characters")

            for url in self.urls:
                data = requests.get(url).content
                filename = os.path.join(self.path, os.path.basename(url))
                with open(filename, "wb") as file:
                    file.write(data)

    # Process, handle one-hot conversion of data etc

    def calculate_stats(self):
        df = self.original_df.copy()
        z_scale_cols = self.meta["num_columns"]["zscaler"]
        self.mean_dict = {}
        self.std_dict = {}
        for c in z_scale_cols:
            # z-score normalization
            self.mean_dict[c] = df[c].mean()
            self.std_dict[c] = df[c].std()

        # Take note of columns to scale with min-max normalization
        # minmax_scale_cols = ["age",  "hours-per-week", "education-num"]
        minmax_scale_cols = self.meta["num_columns"]["minmaxscaler"]
        self.min_dict = {}
        self.max_dict = {}
        for c in minmax_scale_cols:
            self.min_dict[c] = df[c].min()
            self.max_dict[c] = df[c].max()

        self.unique_values_dict = {}
        for c in self.meta["cat_columns"] + [self.meta["y_column"]]:
            self.unique_values_dict[c] = df[c].unique().tolist()

    def one_hot_y(self):
        y = self.original_df[self.meta["y_column"]].apply((lambda x: self.y_mapping_dict[x])).to_numpy().reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(y)
        self.y_enc = enc

        # return enc.transform(y).toarray(), enc

    def one_hot_sensitive(self):
        self.sensitive_mapping_dict = {
            self.meta['sensitive_positive']: True
        }
        for v in self.meta['sensitive_values']:
            if v != self.meta['sensitive_positive']:
                self.sensitive_mapping_dict[v] = False
        sensitive = self.original_df[self.meta["sensitive_column"]].apply((lambda x: self.sensitive_mapping_dict[x])).to_numpy().reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(sensitive)
        self.sensitive_enc = enc

        # return enc.transform(sensitive).toarray(), enc


    
    def process_df(self, df):
        # df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
        # df[self.meta["y_column"]] = df[self.meta["y_column"]].apply((lambda x: 1 if self.meta["y_positive"] in x else 0))
        df[self.meta["y_column"]] = df[self.meta["y_column"]].apply((lambda x: self.y_mapping_dict[x]))

        def oneHotCatVars(x, colname):
            df_1 = x.drop(columns=colname, axis=1)
            df_2 = pd.get_dummies(x[colname], prefix=colname, prefix_sep='_')
            return (pd.concat([df_1, df_2], axis=1, join='inner'))

        colnames = self.meta["cat_columns"]
        # Drop columns that do not help with task
        # df = df.drop(columns=self.dropped_cols, axis=1)
        # Club categories not directly relevant for property inference
        # df["race"] = df["race"].replace(
        #     ['Asian-Pac-Islander', 'Amer-Indian-Eskimo'], 'Other')
        for colname in colnames:
            df = oneHotCatVars(df, colname)

        # Take note of columns to scale with Z-score
        # z_scale_cols = ["fnlwgt", "capital-gain", "capital-loss"]
        z_scale_cols = self.meta["num_columns"]["zscaler"]
        for c in z_scale_cols:
            # z-score normalization
            # df[c] = (df[c] - df[c].mean()) / df[c].std()
            df[c] = (df[c] - self.mean_dict[c]) / self.std_dict[c]

        # Take note of columns to scale with min-max normalization
        # minmax_scale_cols = ["age",  "hours-per-week", "education-num"]
        minmax_scale_cols = self.meta["num_columns"]["minmaxscaler"]
        for c in minmax_scale_cols:
            # z-score normalization
            # df[c] = (df[c] - df[c].min()) / df[c].max()?
            df[c] = (df[c] - self.min_dict[c]) / self.max_dict[c]
        # Drop features pruned via feature engineering
        # prune_feature = [
        #     "workClass:Never-worked",
        #     "workClass:Without-pay",
        #     "occupation:Priv-house-serv",
        #     "occupation:Armed-Forces"
        # ]
        # df = df.drop(columns=prune_feature, axis=1)
        return df
    
    def get_random_data(self, num):
        orig_data = self.original_df.copy()
        orig_columns = orig_data.columns
        # unique_val_dict = {}
        # for col in orig_columns:
        #     unique_val_dict[col] = orig_data[col].unique()
        #     if col not in self.meta['dummy_columns'] and col != self.meta['y_column']:
        #         unique_val_dict[col] = [np.min(unique_val_dict[col]), np.max(unique_val_dict[col])]
        # print(unique_val_dict)
        x = {}
        for col in orig_columns:
            # if col in self.meta['dummy_columns'] or col == self.meta['y_column']:
            if col in self.meta["cat_columns"] or col == self.meta["y_column"]:
                # print()
                np.random.seed(42)
                # x[col] = np.random.choice(unique_val_dict[col].tolist(), num, replace=True)
                x[col] = np.random.choice(self.unique_values_dict[col], num, replace=True)
            elif col in self.meta["num_columns"]["minmaxscaler"]:
                np.random.seed(42)
                # print(unique_val_dict[col][0], unique_val_dict[col][1])
                # x[col] = [random.randint(unique_val_dict[col][0], unique_val_dict[col][1])] * 2
                # x[col] = np.random.randint(unique_val_dict[col][0], unique_val_dict[col][1], size=num)
                x[col] = np.random.uniform(self.min_dict[col], self.max_dict[col], size=num)
                # x[col] = 0
            elif col in self.meta["num_columns"]["zscaler"]:
                x[col] = np.random.normal(self.mean_dict[col], self.std_dict[col], size=num)

        # x = pd.DataFrame([x], columns = orig_data.columns)
        x = pd.DataFrame.from_dict(x)
        # print(x)
        # xp = x.copy()
        # x = pd.get_dummies(x, columns=self.meta['dummy_columns'])
        # x = pd.DataFrame(x, columns=one_hot_columns)
        # x.fillna(0, inplace=True)
        # numerical_features = self.meta['numeric_columns']
        # x[numerical_features] = scaler.transform(x[numerical_features])

        xp = x.copy()
        xp = self.process_df(xp)

        return x, xp

    # Return data with desired property ratios
    def get_x_y(self, P):
        # Scale X values
        # Y = P['income'].to_numpy()
        Y = P[self.meta["y_column"]].to_numpy()
        # X = P.drop(columns='income', axis=1)
        X = P.drop(columns=self.meta["y_column"], axis=1)
        cols = X.columns
        X = X.to_numpy()
        return (X.astype(float), np.expand_dims(Y, 1), cols)

    def get_data(self, split, prop_ratio, filter_prop, custom_limit=None):

        def prepare_one_set(TRAIN_DF, TEST_DF):
            # Apply filter to data
            TRAIN_DF = self.get_filter(TRAIN_DF, filter_prop,
                                  split, prop_ratio, is_test=0,
                                  custom_limit=custom_limit)
            TEST_DF = self.get_filter(TEST_DF, filter_prop,
                                 split, prop_ratio, is_test=1,
                                 custom_limit=custom_limit)

            (x_tr, y_tr, cols), (x_te, y_te, cols) = self.get_x_y(
                TRAIN_DF), self.get_x_y(TEST_DF)

            return (x_tr, y_tr), (x_te, y_te), cols

        if split == "all":
            return prepare_one_set(self.train_df, self.test_df)
        if split == "victim":
            return prepare_one_set(self.train_df_victim, self.test_df_victim)
        return prepare_one_set(self.train_df_adv, self.test_df_adv)

    # Create adv/victim splits, normalize data, etc
    def load_data(self, test_ratio, random_state=42, sampling_condition_dict_list=None):
        # Load train, test data
        # train_data = pd.read_csv(os.path.join(self.path, 'adult.data'),
        #                          names=self.columns, sep=' *, *',
        #                          na_values='?', engine='python')
        # test_data = pd.read_csv(os.path.join(self.path, 'adult.test'),
        #                         names=self.columns, sep=' *, *', skiprows=1,
        #                         na_values='?', engine='python')
        if "train_data_path" in self.meta and "test_data_path" in self.meta:
            train_data = pd.read_csv(self.meta["train_data_path"])
            test_data = pd.read_csv(self.meta["test_data_path"])
        elif sampling_condition_dict_list is not None:
            data = pd.read_csv(self.meta["data_path"])
            sampled_data = None
            for condition_dict in sampling_condition_dict_list:
                sampled_data = pd.concat([sampled_data, data[condition_dict['condition'](data)].sample(n=condition_dict['sample_size'], random_state=random_state)])
            # split into train/test
            train_data, test_data = train_test_split(sampled_data, test_size=test_ratio, random_state=random_state)
        else:
            data = pd.read_csv(self.meta["data_path"])
            # randomly sample 100,000 data points
            data = data.sample(n=100000, random_state=random_state)
            # split into train/test
            train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=random_state)

        # Add field to identify train/test, process together
        train_data['is_train'] = 1
        test_data['is_train'] = 0
        # concat train and test data and reset index
        df = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        if 'transformations' in self.meta:
            for transformation in self.meta['transformations']:
                df = transformation(df)

        self.original_df = df.copy()

        self.one_hot_sensitive()
        self.one_hot_y()

        self.calculate_stats()
        df = self.process_df(df)

        self.df = df.copy()

        # Split back to train/test data
        self.train_df, self.test_df = df[df['is_train']
                                         == 1], df[df['is_train'] == 0]

        # Drop 'train/test' columns
        self.train_df = self.train_df.drop(columns=['is_train'], axis=1)
        self.test_df = self.test_df.drop(columns=['is_train'], axis=1)

        # def s_split(this_df, rs=random_state):
        #     sss = StratifiedShuffleSplit(n_splits=1,
        #                                  test_size=test_ratio,
        #                                  random_state=rs)
        #     # Stratification on the properties we care about for this dataset
        #     # so that adv/victim split does not introduce
        #     # unintended distributional shift
        #     splitter = sss.split(
        #         this_df, this_df[["sex:Female", "race:White", "income"]])
        #     split_1, split_2 = next(splitter)
        #     return this_df.iloc[split_1], this_df.iloc[split_2]

        # # Create train/test splits for victim/adv
        # self.train_df_victim, self.train_df_adv = s_split(self.train_df)
        # self.test_df_victim, self.test_df_adv = s_split(self.test_df)


    def get_filter(self, df, filter_prop, split, ratio, is_test, custom_limit=None):
        if filter_prop == "none":
            return df
        elif filter_prop == "sex":
            def lambda_fn(x): return x['sex:Female'] == 1
        elif filter_prop == "race":
            def lambda_fn(x): return x['race:White'] == 1
        prop_wise_subsample_sizes = {
            "adv": {
                "sex": (1100, 500),
                "race": (2000, 1000),
            },
            "victim": {
                "sex": (1100, 500),
                "race": (2000, 1000),
            },
        }

        if custom_limit is None:
            subsample_size = prop_wise_subsample_sizes[split][filter_prop][is_test]
        else:
            subsample_size = custom_limit
        return heuristic(df, lambda_fn, ratio,
                            subsample_size, class_imbalance=3,
                            n_tries=100, class_col='income',
                            verbose=False)
    
    def get_attack_df(self):
        df = self.df.copy()

        df = df[df['is_train'] == 1]

        meta = self.meta
        # cols_to_drop = [meta["sensitive_column"] + "_" + x for x in meta["sensitive_values"] if x != meta["sensitive_positive"]]
        cols_to_drop = [f'{meta["sensitive_column"]}_{x}' for x in meta["sensitive_values"] if x != meta["sensitive_positive"]]

        df = df.drop(columns=cols_to_drop, axis=1)

        df[meta["y_column"]] = df[meta["y_column"]].apply((lambda x: meta["y_values"][x]))

        def oneHotCatVars(x, colname):
            df_1 = x.drop(columns=colname, axis=1)
            df_2 = pd.get_dummies(x[colname], prefix=colname, prefix_sep='_')
            return (pd.concat([df_1, df_2], axis=1, join='inner'))       
        
        df = oneHotCatVars(df, meta["y_column"])

        self.attack_df = df

        # df.rename(columns={meta["sensitive_column"] + "_" + meta["sensitive_positive"]: meta["sensitive_column"]}, inplace=True)
        df.rename(columns={f'{meta["sensitive_column"]}_{meta["sensitive_positive"]}': meta["sensitive_column"]}, inplace=True)

        X = df.drop(columns = [meta["sensitive_column"], "is_train"])
        y = df[meta["sensitive_column"]]

        self.X_attack, self.y_attack = X, y

        return X, y



# def cal_q(df, condition):
#     qualify = np.nonzero((condition(df)).to_numpy())[0]
#     notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
#     return len(qualify), len(notqualify)


# def get_df(df, condition, x):
#     qualify = np.nonzero((condition(df)).to_numpy())[0]
#     np.random.shuffle(qualify)
#     return df.iloc[qualify[:x]]


# def cal_n(df, con, ratio):
#     q, n = cal_q(df, con)
#     current_ratio = q / (q+n)
#     # If current ratio less than desired ratio, subsample from non-ratio
#     if current_ratio <= ratio:
#         if ratio < 1:
#             nqi = (1-ratio) * q/ratio
#             return q, nqi
#         return q, 0
#     else:
#         if ratio > 0:
#             qi = ratio * n/(1 - ratio)
#             return qi, n
#         return 0, n

# Wrapper for easier access to dataset
class CensusWrapper:
    def __init__(self, filter_prop="none", ratio=0.5, split="all", name="Adult", sensitive_column="default", preload=True, sampling_condition_dict_list=None):
        self.name = name
        self.ds = CensusIncome(name=name, sensitive_column=sensitive_column, preload=preload, sampling_condition_dict_list=sampling_condition_dict_list)
        self.split = split
        self.ratio = ratio
        self.filter_prop = filter_prop

    def load_data(self, custom_limit=None):
        return self.ds.get_data(split=self.split,
                                prop_ratio=self.ratio,
                                filter_prop=self.filter_prop,
                                custom_limit=custom_limit
                                )
    
def oneHotCatVars(x, colname):
    df_1 = x.drop(columns=colname, axis=1)
    df_2 = pd.get_dummies(x[colname], prefix=colname, prefix_sep='_')
    return (pd.concat([df_1, df_2], axis=1, join='inner'))  

def normalize(x_n_tr, x_n_te, normalize=True):
    if normalize:
        # x_n_tr_mean = x_n_tr.mean(axis=0)
        x_n_tr_std = x_n_tr.std(axis=0)
        # delete columns with std = 0
        x_n_tr = x_n_tr.loc[:, x_n_tr_std != 0]
        x_n_te = x_n_te.loc[:, x_n_tr_std != 0]
        x_n_tr_mean = x_n_tr.mean(axis=0)
        x_n_tr_std = x_n_tr.std(axis=0)

        x_n_tr = (x_n_tr - x_n_tr_mean) / x_n_tr_std
        x_n_te = (x_n_te - x_n_tr_mean) / x_n_tr_std
    return x_n_tr, x_n_te

def filter(df, condition, ratio, verbose=True):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    current_ratio = len(qualify) / (len(qualify) + len(notqualify))
    # If current ratio less than desired ratio, subsample from non-ratio
    if verbose:
        print("Changing ratio from %.2f to %.2f" % (current_ratio, ratio))
    if current_ratio <= ratio:
        np.random.shuffle(notqualify)
        if ratio < 1:
            nqi = notqualify[:int(((1-ratio) * len(qualify))/ratio)]
            return pd.concat([df.iloc[qualify], df.iloc[nqi]])
        return df.iloc[qualify]
    else:
        np.random.shuffle(qualify)
        if ratio > 0:
            qi = qualify[:int((ratio * len(notqualify))/(1 - ratio))]
            return pd.concat([df.iloc[qi], df.iloc[notqualify]])
        return df.iloc[notqualify]
    

def heuristic(df, condition, ratio,
              cwise_sample,
              class_imbalance=2.0,
              n_tries=1000,
              class_col="label",
              verbose=True):
    vals, pckds = [], []
    iterator = range(n_tries)
    if verbose:
        iterator = tqdm(iterator)
    for _ in iterator:
        pckd_df = filter(df, condition, ratio, verbose=False)

        # Class-balanced sampling
        zero_ids = np.nonzero(pckd_df[class_col].to_numpy() == 0)[0]
        one_ids = np.nonzero(pckd_df[class_col].to_numpy() == 1)[0]

        # Sub-sample data, if requested
        if cwise_sample is not None:
            if class_imbalance >= 1:
                zero_ids = np.random.permutation(
                    zero_ids)[:int(class_imbalance * cwise_sample)]
                one_ids = np.random.permutation(
                    one_ids)[:cwise_sample]
            else:
                zero_ids = np.random.permutation(
                    zero_ids)[:cwise_sample]
                one_ids = np.random.permutation(
                    one_ids)[:int(1 / class_imbalance * cwise_sample)]

        # Combine them together
        pckd = np.sort(np.concatenate((zero_ids, one_ids), 0))
        pckd_df = pckd_df.iloc[pckd]

        vals.append(condition(pckd_df).mean())
        pckds.append(pckd_df)

        # Print best ratio so far in descripton
        if verbose:
            iterator.set_description(
                "%.4f" % (ratio + np.min([np.abs(zz-ratio) for zz in vals])))

    vals = np.abs(np.array(vals) - ratio)
    # Pick the one closest to desired ratio
    picked_df = pckds[np.argmin(vals)]
    return picked_df.reset_index(drop=True)


def filter_random_data(ds, clf, confidence_threshold=0.99, num_samples=100000, seed=42):
    def oneHotCatVars(x, colname):
        df_1 = x.drop(columns=colname, axis=1)
        df_2 = pd.get_dummies(x[colname], prefix=colname, prefix_sep='_')
        return (pd.concat([df_1, df_2], axis=1, join='inner'))

    random_df, random_oh_df = ds.ds.get_random_data(num_samples)
    data_dict = ds.ds.meta
    X_random = random_oh_df.drop(data_dict['y_column'], axis=1)
    default_cols = X_random.columns
    # y = random_oh_df[data_dict['y_column']]

    sensitive_attr = data_dict['sensitive_column']
    sensitive_values = data_dict['sensitive_values']
    prediction_columns = []
    for sensitive_value in sensitive_values:
        X_random[sensitive_attr + "_" + sensitive_value] = pd.Series(np.ones(X_random.shape[0]), index=X_random.index)
        for other_value in sensitive_values:
            if other_value != sensitive_value:
                X_random[sensitive_attr + "_" + other_value] = pd.Series(np.zeros(X_random.shape[0]), index=X_random.index)

        newcolname = "prediction_" + sensitive_value
        prediction_columns.append(newcolname)
        X_random[newcolname] = pd.Series(np.argmax(clf.predict(X_random[default_cols]), axis=1), index=X_random.index)
        X_random["confidence_" + sensitive_value] = pd.Series(np.max(clf.predict_proba(X_random[default_cols]), axis=1), 
                                                            index=X_random.index)

    X_random['all_predictions'] = X_random[prediction_columns].apply(pd.Series.unique, axis=1)

    # X_random = X_random[X_random['all_predictions'].apply(lambda x: len(x)==len(sensitive_values))]


    dfs = []

    for sensitive_value in sensitive_values:
        X_temp = X_random[default_cols].copy()

        X_temp[sensitive_attr + "_" + sensitive_value] = pd.Series(np.ones(X_temp.shape[0]), index=X_random.index)
        for other_value in sensitive_values:
            if other_value != sensitive_value:
                X_temp[sensitive_attr + "_" + other_value] = pd.Series(np.zeros(X_temp.shape[0]), index=X_random.index)

        X_temp[data_dict['y_column']] = X_random["prediction_" + sensitive_value].apply(lambda x: data_dict['y_values'][x])
        X_temp['confidence'] = X_random["confidence_" + sensitive_value]

        dfs.append(X_temp.copy())

    random_oh_df = pd.concat(dfs, ignore_index=True)

    random_oh_df = random_oh_df[random_oh_df['confidence'].apply(lambda x: x > confidence_threshold)].drop('confidence', axis=1)

    random_oh_df = oneHotCatVars(random_oh_df, data_dict['y_column'])

    #convert datatype to float
    random_oh_df = random_oh_df.astype(float)

    return random_oh_df

    


