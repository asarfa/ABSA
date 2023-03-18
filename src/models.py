from transformers import AutoConfig, AutoModel, get_scheduler
import torch
import torch.nn as nn
import numpy as np
import time
import os

from data import myDataLoader, myDataset
from utils import cat_to_label
from sklearn.metrics import f1_score


class ABSAClassifierBase(nn.Module):

    def __init__(self, model_name: str, dropout: float = 0.1):
        super(ABSAClassifierBase, self).__init__()
        self.lmconfig = AutoConfig.from_pretrained(model_name)
        self.lm = AutoModel.from_pretrained(model_name)
        self.emb_dim = self.lmconfig.hidden_size
        self.output_size = 3
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.emb_dim, self.output_size),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.lm(input_ids, attention_mask, token_type_ids, return_dict=False)
        logits = self.classifier(pooled_output)
        return logits


class EarlyStopping:
    """
    This class allows to stop the training when the validation loss doesn't decrease anymore
    """

    def __init__(
            self,
            tolerance: int = 5):

        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.train_acc, self.best_val_acc, self.val_f1 = -np.inf, -np.inf, -np.inf

    def __call__(self, train_acc, val_acc, val_f1):
        if round(val_acc, 3) <= round(self.best_val_acc, 3):
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.train_acc = train_acc
            self.best_val_acc = val_acc
            self.val_f1 = val_f1
            self.counter = 0


class ABSAClassifier:
    """
    This class allows to compute the main methods to fit and evaluate a model and predict output on testing set
    """

    def __init__(self, model_name: str = None, dataset: myDataset = None, device = None, save: bool = True,
                 epochs: int = 20, lr: float = 2e-5, l2reg: float = 0.01,
                 batch_size: int = 16, dropout: float = 0.1, seed: int = 42, verbose: bool = True,
                 criterion=nn.CrossEntropyLoss()):
        self.device = device
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.l2reg = l2reg
        self.verbose = verbose
        self.best_epoch = None
        self.__set_seed(seed)
        self.__instantiate_dataloader(dataset, batch_size)
        self.__instantiate_save(save, model_name, batch_size, dropout)
        self.__instantiate_model(model_name, dropout)
        self.__instantiate_optimizer()
        #self.__instantiate_lrscheduler()

    def to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device, non_blocking=True)

    def __set_seed(self, seed):
        if seed is not None: torch.manual_seed(seed)

    def __instantiate_save(self, save, name, batch_size, dropout):
        self.name = name + '_lr' + str(self.lr) + '_batchsize' + str(batch_size) + '_dropout' + str(dropout) + '_l2reg' + str(self.l2reg)
        self.save = save
        if self.save:
            self.path = os.path.join('\savings', self.name)
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def __instantiate_dataloader(self, dataset, batch_size):
        self.dataloader = myDataLoader(dataset, batch_size)

    def __instantiate_model(self, model_name, dropout):
        self.model = self.to_device(ABSAClassifierBase(model_name, dropout))

    def __instantiate_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.l2reg)

    """
    def __instantiate_lrscheduler(self):
        num_update_steps_per_epoch = len(self.dataloader.train)
        num_training_steps = self.epochs * num_update_steps_per_epoch

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
    """

    @staticmethod
    def __logits_to_pred(outputs):
        pred = torch.argmax(torch.exp(torch.log_softmax(outputs, dim=1)), dim=1)
        return pred

    def __accuracy(self, output, true):
        pred = self.__logits_to_pred(output)
        correct = torch.eq(pred.cpu(), true.cpu()).int()
        return float(correct.sum()) / float(correct.numel())

    def __f1score(self, output, true):
        pred = self.__logits_to_pred(output)
        return f1_score(true.cpu(), pred.cpu(), labels=[0, 1, 2], average='macro', zero_division=0)

    def __train_model(self, dataloader):
        # set the model in training mode
        self.model.train()
        # stores the loss
        train_losses, train_accuracy = [], []
        for batch in dataloader:
            # send input to device
            input_ids, attention_mask, token_type_ids, y = self.to_device(batch)
            # zero out previous accumulated gradients
            self.optimizer.zero_grad()
            # perform forward pass and calculate accuracy + loss
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            loss = self.criterion(outputs, y.long())
            # perform backpropagation and update model parameters
            loss.backward()
            self.optimizer.step()
            #self.lr_scheduler.step()

            train_losses.append(loss.item())
            train_accuracy.append(self.__accuracy(outputs, y))
        return np.mean(train_losses), np.mean(train_accuracy)

    @torch.no_grad()
    def __evaluate_model(self, dataloader):
        # Allows to evaluate on dataloader or predict on datalaoder
        # set the model in eval mode
        self.model.eval()
        losses, accuracy, f1score, predictions = [], [], [], []
        for batch in dataloader:
            # send input to device
            input_ids, attention_mask, token_type_ids, y = self.to_device(batch)
            # perform forward pass and calculate accuracy + loss
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            loss = self.criterion(outputs, y.long())

            predictions.append(self.__logits_to_pred(outputs).cpu().detach().numpy())
            losses.append(loss.item())
            accuracy.append(self.__accuracy(outputs, y))
            f1score.append(self.__f1score(outputs, y))

        return np.mean(losses), np.mean(accuracy), np.mean(f1score), predictions

    def __compute_early_stopping(self, epoch, my_es, train_acc_mean, val_acc_mean, val_f1_mean):
        break_it = False
        my_es(train_acc_mean, val_acc_mean, val_f1_mean)
        if my_es.early_stop:
            if self.verbose:
                print(
                    f'At last epoch {epoch}, the early stopping tolerance = {my_es.tolerance} has been reached,'
                    f' the acc of validation is not increasing anymore -> stop it')
            break_it = True
        return my_es, break_it

    def __compute_verbose_train(self, epoch, start_time, train_loss_mean, val_loss_mean, train_acc_mean, val_acc_mean, val_f1_mean):
        print(
            "Epoch [{}] took {:.2f}s | train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f}".format(
                epoch, time.time() - start_time, train_loss_mean, train_acc_mean, val_loss_mean, val_acc_mean, val_f1_mean))

    def fit(self):
        my_es = EarlyStopping()
        if self.verbose: print(f'********************************* Training of {self.name} *********************************')
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss_mean, train_acc_mean = self.__train_model(self.dataloader.train)
            val_loss_mean, val_acc_mean, val_f1_mean, _ = self.__evaluate_model(self.dataloader.eval)

            if self.verbose:
                self.__compute_verbose_train(epoch, start_time, train_loss_mean, val_loss_mean, train_acc_mean,
                                             val_acc_mean, val_f1_mean)

            my_es, break_it = self.__compute_early_stopping(epoch, my_es, val_loss_mean, val_acc_mean, val_f1_mean)
            if break_it:
                break
            # to be able to restore the best weights with the best epoch
            if my_es.counter == 0: self.__save_args(epoch)

        if break_it:
            self.best_epoch = epoch - my_es.tolerance
        else:
            self.best_epoch = self.epochs

        self.train_acc, self.val_acc, self.val_f1 = my_es.train_acc, my_es.best_val_acc, my_es.val_f1
        return self.train_acc, self.val_acc, self.val_f1

    def predict(self, dataset: myDataset):
        batch_size = 1
        dataloader = myDataLoader(dataset, batch_size)
        if self.save:
            self.__load_args()
            self.__delete_files()
        loss_mean, acc_mean, f1_mean, predictions = self.__evaluate_model(dataloader.eval)
        predictions = list(map(lambda pred: cat_to_label(pred), predictions))
        if self.verbose: print(f'The testing acc is {acc_mean}, the testing f1 is {f1_mean}')
        return predictions

    def __save_args(self, epoch: int):
        torch.save({"model_state_dict": self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    f'{self.path}/model_{epoch}.pt')

    def __find_best_epoch(self):
        try:
            epoch = max([int(f.split('_')[1].split('.')[0]) for f in os.listdir(self.path)])
        except ValueError:
            epoch = None
        return epoch

    def __load_args(self):
        epoch = self.__find_best_epoch()
        path_ = f'{self.path}/model_{epoch}.pt'
        if os.path.exists(path_):
            if self.device.type == 'cpu':
                checkpoint = torch.load(path_, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(path_)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def __delete_files(self):
        args_file = os.listdir(self.path)
        for file in args_file:
            delete_filename = os.path.join(self.path, file)
            open(delete_filename, 'w').close()
            os.remove(delete_filename)