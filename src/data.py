import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from utils import process_data
from analysis import AnalysisData


class ABSADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.labels = torch.tensor(data['label'].to_list())
        input_ids, token_type_ids = [], []
        for i in range(len(data)):
            sentence_ids = tokenizer.text_to_ids(data['clean_sentence'].iloc[i])
            aspect_ids = tokenizer.text_to_ids(data['clean_target_term'].iloc[i])
            input_ids.append([101]+sentence_ids+[102]+aspect_ids+[102])
            token_type_ids.append([0] * (len(sentence_ids) + 2) + [1] * (len(aspect_ids) + 1))
        self.max_seq_len = max([len(ids) for ids in input_ids])
        input_ids, attention_mask = tokenizer.pad_trunc(input_ids, self.max_seq_len, attention=True)
        self.input_ids = torch.tensor(input_ids)
        self.attention_mask = torch.tensor(attention_mask)
        self.token_type_ids = torch.tensor(tokenizer.pad_trunc(token_type_ids, self.max_seq_len))
        self.category = torch.tensor(pd.get_dummies(data['aspect_category']).values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.attention_mask[i], self.token_type_ids[i], self.labels[i]


class myDataset:
    """
    This class allows to create the datasets for both training and validation
    """

    def __init__(self, tokenizer, train_filename: str = None, dev_filename: str = None, verbose: bool = False):
        self.set(tokenizer, train_filename, dev_filename, verbose)

    def get_dataset(self, tokenizer, filename, verbose):
        """
        Instanciation of trainset with augmentation, preprocessing and normalization transformation
        And the val set
        """
        if filename is not None:
            data = process_data(filename)
            if verbose:
                print(f"*************** Analysis of preprocessed {filename} ***************")
                print(AnalysisData(data).__dict__)
            if filename == '../data/devdata.csv':
                #from soa_models import InstructABSA
                #InstructABSA(data)
                pass
            return ABSADataset(data=data, tokenizer=tokenizer)
        else:
            return None

    def set(self, tokenizer, train_filename: str = None, dev_filename: str = None, verbose: bool = None):
        """
        Instanciation of train dataset and val dataset
        """
        self.train = self.get_dataset(tokenizer, train_filename, verbose)
        self.eval = self.get_dataset(tokenizer, dev_filename, verbose)


class myDataLoader:
    """
    This class allows to create the dataloaders for both training and validation
    """

    def __init__(self, mydataset: myDataset, batch_size: int = 16):
        self.batch_size = batch_size
        self.set(mydataset)

    def set(self, mydataset: myDataset):
        """
        Instanciation of train dataloader and val dataloader
        """
        if mydataset.train is not None:
            self.train = DataLoader(mydataset.train, shuffle=True,
                                               batch_size=self.batch_size, num_workers=2)
        if mydataset.eval is not None:
            self.eval = DataLoader(mydataset.eval, shuffle=False,
                                               batch_size=self.batch_size, num_workers=2)