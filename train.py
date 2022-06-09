import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar
from sklearn import metrics

from utils.utils import weights_init_normal
from model import autoencoder, network
from early_stopping import EarlyStopping

class TrainerDeepSAD:
    def __init__(self, args, train_data, val_data, device, pretrain_data=None, pretrain_val_data=None):
        self.args = args
        self.train_loader = train_data
        self.val_loader = val_data
        self.device = device
        self.pretrain_loader = pretrain_data
        self.pretrain_val_loader = pretrain_val_data
        self.autoencoder = autoencoder
        self.network = network

    def pretrain_validation(self, ae):
        ae.eval()
        val_loss = 0
        with torch.no_grad():
            for x in Bar(self.pretrain_val_loader):
                x = x.float().to(self.device)

                x_hat = ae(x)
                #reconst_loss = criterion(x_hat, x)
                #loss = torch.mean(reconst_loss)
                loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))

                val_loss += loss.item()
         
        loss = val_loss / len(self.pretrain_val_loader)

        print(f"Validation Loss: {loss:.6f}")

        return loss


    def pretrain(self):
        """ Pretraining the weights for the deep SAD network using autoencoder"""
        ae = self.autoencoder(self.args.latent_dim).to(self.device)

        early_stopping = EarlyStopping(patience=self.args.patience_pretrain, verbose=True, path=self.args.ae_save_path, delta=self.args.delta_pretrain)

        #criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=self.args.lr_milestones_ae, gamma=0.1)

        if self.args.resume_pretrain:
            state_dict = torch.load(self.args.ae_progress_save_path)
            ae.load_state_dict(state_dict['net_dict'])
            optimizer.load_state_dict(state_dict['optimizer_dict'])
            scheduler.load_state_dict(state_dict['scheduler_dict'])
            print("completed until {} epoch(loss : {:.3f}).".format(state_dict['epoch'], state_dict['loss']))
        else:
            ae.apply(weights_init_normal)
            log = {'train_loss' : [],
                   'validation_loss' : []}

        for epoch in range(self.args.num_epochs_ae):
            if self.args.resume_pretrain:
                if epoch <= state_dict['epoch']:
                    continue
            ae.train()
            total_loss = 0
            print(f"Pretraining Autoencoder... Epoch: {epoch}")
            for x in Bar(self.pretrain_loader):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = ae(x)
                #reconst_loss = criterion(x_hat, x)
                #loss = torch.mean(reconst_loss)
                #loss.backward()
                loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                loss.backward()
                
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            loss = total_loss / len(self.pretrain_loader)
            
            print(f"Loss: {loss:.6f}")

            val_loss = self.pretrain_validation(ae)

            log['train_loss'].append(loss)
            log['validation_loss'].append(val_loss)

            early_stopping(val_loss, ae, log)

            if early_stopping.early_stop: 
                state_dict = torch.load(self.args.ae_save_path)
                ae.load_state_dict(state_dict['net_dict'])
                self.save_weights_for_DeepSAD(ae, self.pretrain_loader, log)
                break
            
            """
            if epoch % 5 == 0 or epoch == self.args.num_epochs_ae-1:
                torch.save({'epoch': epoch,
                            'net_dict': ae.state_dict(),
                            'optimizer_dict': optimizer.state_dict(),
                            'scheduler_dict': scheduler.state_dict(),
                            'loss': loss
                                }, self.args.ae_progress_save_path)
                print('Saved Progress.')
            """
        
        self.save_weights_for_DeepSAD(ae, self.pretrain_loader, log)


    def save_weights_for_DeepSAD(self, model, dataloader, log):
        """Initialize Deep SAD weights using the encoder weights of the pretrained autoencoder."""
        c = self.set_c(model, dataloader)
        net = self.network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict(),
                    'log': log}, self.args.ae_save_path)
        print(f"pretrained_parameters saved to {self.args.ae_save_path}.")


    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x in Bar(dataloader):
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        print('Setting Center done.')
        return c

    def validation(self, net, center):
        net.eval()
        label_score = []
        val_loss = 0
        with torch.no_grad():
            for x, labels in Bar(self.val_loader):
                x, labels = x.float().to(self.device), labels.to(self.device)

                z = net(x)
                dist = torch.sum((z - center) ** 2, dim=1)
                losses = torch.where(labels == 0, dist, self.args.eta * ((dist + self.args.eps) ** -labels.float()))
                loss = torch.mean(losses)
                scores = dist

                val_loss += loss.item()
                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))
         
        loss = val_loss / len(self.val_loader)

        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
        auc = metrics.auc(recall, precision)
        F_measure = 2 * precision * recall / (precision + recall)

        F_max_id = np.argmax(F_measure)
        F_measure_max = np.max(F_measure)
        precision_F_max = precision[F_max_id]
        recall_F_max = recall[F_max_id]
        threshold_F_max = thresholds[F_max_id]

        print(f"Validation Loss: {loss:.3f}")
        print(f"F-measure : {F_measure_max:.4f}  Precision : {precision_F_max:.4f}  Recall : {recall_F_max:.4f}  Threshold : {threshold_F_max:.4f}")
        print(f"PR-AUC : {100.*auc:.2f}%")

        return loss, auc, F_measure_max


    def train(self):
        """Training the Deep SAD model"""
        net = self.network(self.args.latent_dim).to(self.device)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, path=self.args.net_save_path, delta=self.args.delta)
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                               milestones=self.args.lr_milestones, gamma=0.1)

        if self.args.resume_train:
            state_dict = torch.load(self.args.net_progress_save_path)
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
            optimizer.load_state_dict(state_dict['optimizer_dict'])
            scheduler.load_state_dict(state_dict['scheduler_dict'])
            log = state_dict['log']
            print(f"completed until {state_dict['epoch']} epoch(train_loss : {log['train_loss'][-1]:.3f}")
        else:
            state_dict = torch.load(self.args.ae_save_path)
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
            log = {
                'train_loss' : [],
                'validation_loss' : [],
                'PR-AUC' : [],
                'F1' : []
                        }
            print(f"Loaded weights of {self.args.ae_save_path}")

        for epoch in range(self.args.num_epochs):
            if self.args.resume_train and epoch <= state_dict['epoch']:
                continue
            net.train()
            total_loss = 0
            print(f"Training Deep SAD... Epoch : {epoch}")
            for x, semi_targets in Bar(self.train_loader):
                x, semi_targets = x.float().to(self.device), semi_targets.to(self.device)

                optimizer.zero_grad()
                z = net(x)
                dist = torch.sum((z - c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.args.eta * ((dist + self.args.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            loss = total_loss/len(self.train_loader)
            print(f"Training Loss: {loss:.3f}")

            val_loss, auc, f1 = self.validation(net, c)

            log['train_loss'].append(loss)
            log['validation_loss'].append(val_loss)
            log['PR-AUC'].append(auc)
            log['F1'].append(f1)

            early_stopping(val_loss, net, log, center=c)

            if early_stopping.early_stop: 
                break
            """
            if epoch % 5 == 0 or epoch == self.args.num_epochs-1:
                torch.save({'epoch': epoch,
                            'net_dict': net.state_dict(),
                            'center': c.cpu().data.numpy().tolist(),
                            'optimizer_dict': optimizer.state_dict(),
                            'scheduler_dict': scheduler.state_dict(),
                            'log': log,
                            'loss': loss
                                }, self.args.net_progress_save_path)
                print('Saved Progress.')
            

        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict(),
                    'log': log}, self.args.net_save_path)
        print('network_parameters saved to {}.'.format(self.args.net_save_path))
        """
        print('Finish.')
        
