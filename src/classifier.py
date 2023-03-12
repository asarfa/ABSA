from typing import List
import torch
from data import myDataset
from models import ABSATokenizer, ABSAClassifier
from tuning import ABSATuning
import pickle
import os


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """
    def __init__(self):
        self.inner_model: ABSAClassifier = None
        self.name_model: str = None
        self.tokenizer: str = None

    ############################################# comp
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        self.name_model = 'bert-base-cased'
        self.tokenizer = ABSATokenizer(self.name_model)
        dataset = myDataset(self.tokenizer, train_filename, dev_filename)

        path = f'savings/{self.name_model}_best_params.pkl'
        if not os.path.exists(path):
            tuning = ABSATuning(self.name_model, dataset, device)
            tuning.fit()

        with open(path, "rb") as file:
            best_params = pickle.load(file)
        self.inner_model = ABSAClassifier(self.name_model, dataset, device, verbose=True, lr=best_params['lr'],
                               l2reg=best_params['l2reg'], epochs=best_params['epoch'],
                               batch_size=best_params['batch_size'], dropout=best_params['dropout'], seed=None)
        self.inner_model.fit()

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        dataset = myDataset(self.tokenizer, dev_filename=data_filename)
        predictions = self.inner_model.predict(dataset)
        return predictions






