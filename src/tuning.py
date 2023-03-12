import random
from itertools import product
import pandas as pd
import numpy as np
import pickle

from data import myDataset
from models import ABSAClassifier


class ABSATuning:

    def __init__(self, name_model: str, dataset: myDataset, device, n_iter_opt: int = None, verbose: bool = True):
        self.name_model = name_model
        self.dataset = dataset
        self.device = device
        self.n_iter_opt = n_iter_opt
        self.verbose = verbose
        self.metric_to_max = 'acc_val'

    @property
    def hyperparam_space(self):
        return {'lr': [2e-5, 5e-5],
                'l2reg': [0.1, 0.01],
                'dropout': [0.1, 0.2],
                'batch_size': [16, 32, 64]}

    @staticmethod
    def combination_params(space):
        """
        Combinations of all hyper parameters
        """
        params, values = zip(*space.items())
        product_params = list(product(*values))
        dict_param = dict(map(lambda i: (i, dict(zip(params, product_params[i]))),
                              range(len(product_params))))
        return params, product_params, dict_param

    @staticmethod
    def generate_keys(n_iter_opt, product_params: list, dict_param: dict):
        """
        Defining max number of hyperoptimization trials
        """
        if n_iter_opt is not None:
            random_keys = random.sample(list(dict_param), n_iter_opt)
        else:
            random_keys = list(range(len(product_params)))
        return random_keys

    def fit(self):
        params, product_params, dict_param = self.combination_params(self.hyperparam_space)
        random_keys = self.generate_keys(self.n_iter_opt, product_params, dict_param)
        """
        Instantiate DF to save the results
        """
        results = pd.DataFrame(columns=['acc_train', 'acc_val', 'f1_val'] + list(params) + ['best_epoch'],
                               index=range(len(random_keys)))
        best_val_score = -np.inf
        best_params = None
        if self.verbose:
            print('*'*50)
            print(f'Finding best hyperparameters according to space:\n'
                  f'{self.hyperparam_space}'
                  f' resulting in {len(random_keys)} combinations'
                  f'\nSorted according to maximum {self.metric_to_max}')
            print('*'*50)
        for i, key in enumerate(random_keys):
            params = dict_param[key]
            model = ABSAClassifier(self.name_model, self.dataset, self.device, verbose=True, lr=params['lr'], l2reg=params['l2reg'],
                                   batch_size=params['batch_size'], dropout=params['dropout'])
            acc_train, acc_val, f1_val = model.fit()
            if acc_val > best_val_score:
                best_val_score = acc_val
                best_params = params.copy()
                best_params['epoch'] = model.best_epoch
            results.iloc[i] = [acc_train, acc_val, f1_val] + list(params.values()) + [model.best_epoch]
            print(results)
        results = results.sort_values(self.metric_to_max, ascending=False)[:5]
        if self.verbose:
            print(f'Top 5 models leading to the max accuracy on validation set: ')
            print(results.to_string())
            print(f'Best hyper parameters: {best_params}')
        with open(f'savings/{self.name_model}_best_params.pkl', "wb") as file:
            pickle.dump(best_params, file)
