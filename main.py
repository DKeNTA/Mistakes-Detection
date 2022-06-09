import numpy as np
import argparse
import torch
import sys

from preprocess import get_mydata
from train import TrainerDeepSAD
from test import TesterDeepSVDD


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../dataset')
    parser.add_argument('--val_data_dir', type=str, default='../val')
    parser.add_argument('--test_data_dir', type=str, default='../datasets/test/melspectrograms')
    parser.add_argument('--mode', type=str, choices=['svdd', 'sad'], default='sad')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help="number of epochs")
    parser.add_argument('--num_epochs_ae', type=int, default=150,
                        help="number of epochs for the pretraining")
    parser.add_argument('--patience', type=int, default=50,
                        help="Patience for Early Stopping")
    parser.add_argument('--patience_pretrain', type=int, default=50,
                        help="Patience for Early Stopping of Pretrain")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_ae', type=float, default=5e-4,
                        help='learning rate for autoencoder')
    parser.add_argument('--weight_decay', type=float, default=0.5e-6,
                        help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--weight_decay_ae', type=float, default=0.5e-3,
                        help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument('--lr_milestones_ae', type=list, default=[50])
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of the latent variable z')
    parser.add_argument('--delta', type=float, default=0.02)                    
    parser.add_argument('--delta_pretrain', type=float, default=0.1)
    parser.add_argument('--ae_save_path', type=str, default='weights/pretrained_parameters.pth')
    parser.add_argument('--net_save_path', type=str, default='weights/network_parameters.pth')
    parser.add_argument('--ae_progress_save_path', type=str, default='weights/ae_progress_parameters.pth')
    parser.add_argument('--net_progress_save_path', type=str, default='weights/network_progress_parameters.pth')
    parser.add_argument('--pretrain', action='store_true',
                        help='Pretrain the network using an autoencoder')
    parser.add_argument('--resume_pretrain', action='store_true')
    parser.add_argument('--resume_train', action='store_true')

    args = parser.parse_args()

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if args.pretrain:
        pretrain_data = get_mydata(args, mode='pretrain')
        pretrain_val_data = get_mydata(args, mode='pretrain_val')
        train_data = get_mydata(args, mode='train')
        val_data = get_mydata(args, mode='val')

        Network = TrainerDeepSAD(args, train_data, val_data, device, pretrain_data=pretrain_data, pretrain_val_data=pretrain_val_data)

        Network.pretrain()
        Network.train()
    else:
        train_data = get_mydata(args, mode='train')
        val_data = get_mydata(args, mode='val')

        Network = TrainerDeepSAD(args, train_data, val_data, device)

        Network.train()

    """
    if args.mode == 'svdd':
        from train_DeepSVDD import TrainerDeepSVDD
        Network = TrainerDeepSVDD(args, train_data, device)
    else:
        from train_DeepSAD import TrainerDeepSAD
        pretrain_data = get_mydata(args, pretrain=True)
        Network = TrainerDeepSAD(args, pretrain_data, train_data, device)
    """

    test_data = get_mydata(args, mode='test')

    Network_tester = TesterDeepSVDD(args.net_save_path, args.latent_dim, save=True)

    Network_tester.test_dataset(test_data, mapping=False)
