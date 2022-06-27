import numpy as np
import torch
import pickle

import datetime as dt

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model, log, center=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, center, log)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                now = dt.datetime.now()
                log_filename = 'log_{}.txt'.format(now.strftime('%m%d%H%M'))
                f = open(log_filename, 'wb')
                pickle.dump(log, f)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, center, log)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, center, log):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if center == None:
            torch.save({'net_dict': model.state_dict(),
                        'log': log}, self.save_path)
        else:
            torch.save({'center': center.cpu().data.numpy().tolist(),
                        'net_dict': model.state_dict(),
                        'log': log}, self.save_path)
        self.val_loss_min = val_loss