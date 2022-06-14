import argparse
import torch
from torch import optim
import torch.nn.functional as F
from barbar import Bar
import numpy as np
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

from model import Encoder
from preprocess import get_mydata


class TesterDeepSVDD:
    def __init__(self, parameters_path, latent_dim, save=False):
        #self.args = args
        #self.train_loader = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Encoder(latent_dim).to(self.device)
        state_dict = torch.load(parameters_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(state_dict['net_dict'])
        self.c = torch.Tensor(state_dict['center']).to(self.device)
        #print(state_dict['center'])
        self.save = save

        self.net.eval()
        if save:
            self.log = {'Weight Path':parameters_path}

    def evaluation(self, labels, scores):
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
        auc = metrics.auc(recall, precision)
        f1 = 2 * precision * recall / (precision + recall)

        f1_max_idx = np.argmax(f1)

        f1_max = np.max(f1)
        precision_F1_max = precision[f1_max_idx]
        recall_F1_max = recall[f1_max_idx]
        threshold_F1_max = thresholds[f1_max_idx]

        #for p,r,F,t in zip(precision, recall, F_measure, thresholds):
        #    print(f"Precision : {p:.4f}  Recall : {r:.4f}  F_measure : {F:.4f} Thresholds : {t:.4f}")

        print(f"F measure : {f1_max:.4f}\nPrecision : {precision_F1_max:.4f}\nRecall    : {recall_F1_max:.4f}\nThreshold : {threshold_F1_max:.4f}")
        print(f"PR-AUC    : {100.*auc:.2f}%")

        if self.save:
            result = {'PR-AUC':auc, 'F1':f1_max, 'Precision':precision_F1_max, 'Recall':recall_F1_max, 'Threshold':threshold_F1_max}
            self.log.update(result)
            with open('result.csv', 'a') as f:
                fieldnames = ['Weights Path', 'PR-AUC', 'F1', 'Precision', 'Recall', 'Threshold']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(self.log)

        """
        plt.plot(recall, precision, label='PR curve (area = %.4f)'%auc)
        plt.legend()
        plt.title('PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        plt.grid()
        plt.show()
        """
       

    def test_dataset(self, data_loader, mapping=True):
        label_score = []
        label_z = []
        print('Testing...')
        with torch.no_grad():
            for x, labels in Bar(data_loader):
                x, labels = x.float().to(self.device), labels.to(self.device)

                z = self.net(x)
                dist = torch.sum((z - self.c) ** 2, dim=1)
                scores = dist

                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))
                #scores.append(score.detach().cpu())
                label_z += list(zip(labels.cpu().data.numpy().tolist(),
                                   z.cpu().data.numpy().tolist()))


        labels, scores = zip(*label_score)
        labels = (label if label==0 else 1 for label in list(labels))
        #labels = (label if label<2 else 0 for label in list(labels))
        labels = np.array(tuple(labels))
        scores = np.array(scores)

        self.evaluation(labels, scores)

        if mapping:
            label_num = 5

            c_label = [-1]
            label_z += list(zip(c_label,
                               [self.c.cpu().data.numpy().tolist()]))

            z_labels, z = zip(*label_z)
            z_labels = np.array(z_labels)
            z = np.array(z)

            tsne_2d = TSNE(n_components = 2, random_state=54) # n_componentsは低次元データの次元数
            #tsne_3d = TSNE(n_components = 3)
            z_2d = tsne_2d.fit_transform(z)
            #z_3d = tsne_3d.fit_transform(z)

            colors = ['blue', 'red', 'green', 'orange', 'purple', 'pink', 'lightblue', 'black']
            markers = ['o', 'v', '^', 's', 'x', 'D']
            labels = ['normal', 'buzz', 'mid-buzz', 'muffled', 'mute', 'others', 'tmp', 'center']


            fig = plt.figure(figsize=(12, 8))
            fig.suptitle('Feature space Plot')

            ax1 = fig.add_subplot(1, 1, 1)
            for i in range(-1, label_num):
                target = z_2d[z_labels == i]
                ax1.scatter(target[:, 0], target[:, 1], alpha=0.75 if colors[i] != 'black' else 1.0, color=colors[i], label=labels[i], marker=markers[i])
            plt.legend(fontsize=16)
            ax1.set_title('Feature Space (2D) Plot')

            """
            #fig = plt.figure(figsize=(8, 8)).gca(projection='3d')
            ax2 = fig.add_subplot(1, 1, 1, projection='3d')
            for i in range(-1, label_num):
                target = z_3d[z_labels == i]
                ax2.scatter(target[:, 0], target[:, 1], target[:, 2], label=str(i), alpha=0.75, color=colors[i])
            #plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
            ax2.set_title('3D Plot')
            """

            plt.show()

    def test_dataset_PN(self, data_loader, th):
        label_score_z = []
        print('Testing...')
        with torch.no_grad():
            for x, labels in Bar(data_loader):
                x, labels = x.float().to(self.device), labels.to(self.device)

                z = self.net(x)
                dist = torch.sum((z - self.c) ** 2, dim=1)
                scores = dist

                label_score_z += list(zip(labels.cpu().data.numpy().tolist(),
                                          scores.cpu().data.numpy().tolist(),
                                          z.cpu().data.numpy().tolist()))
                #scores.append(score.detach().cpu())
                #label_z += list(zip(labels.cpu().data.numpy().tolist(),
                #                   z.cpu().data.numpy().tolist()))

        c_label = [-1]
        c_score = [0]
        label_score_z += list(zip(c_label,
                                  c_score,
                                  [self.c.cpu().data.numpy().tolist()]))
        labels, scores, z = zip(*label_score_z)
        labels_PN = []
        for label, score in zip(labels, scores):
            if label == -1:
                label_PN = -1
            elif score < th:
                if label == 0:
                    label_PN = 0
                else:
                    label_PN = 2
            else:
                if label == 0:
                    label_PN = 3
                else:
                    label_PN = 1

            labels_PN.append(label_PN)

        labels_PN = np.array(tuple(labels_PN))
        #scores = np.array(scores)
        z = np.array(z)

        #self.evaluation(labels_PN, scores)

        tsne_2d = TSNE(n_components = 2, random_state=54) # n_componentsは低次元データの次元数
        #tsne_3d = TSNE(n_components = 3)
        z_2d = tsne_2d.fit_transform(z)
        #z_3d = tsne_3d.fit_transform(z)

        colors = ['blue', 'red', 'green', 'orange', 'black']
        markers = ['o', 'x', '^', 's', 'v', 'D']
        labels = ['TN', 'TP', 'FN', 'FP', 'center']

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Feature space Plot')

        ax1 = fig.add_subplot(1, 1, 1)
        for i in range(-1, 4):
            target = z_2d[labels_PN == i]
            ax1.scatter(target[:, 0], target[:, 1], alpha=0.75 if colors[i] != 'black' else 1.0, color=colors[i], label=labels[i], marker=markers[i])
        plt.legend(fontsize=16)
        ax1.set_title('Feature Space (2D) Plot')

        plt.show()

    def test_data(self, data):
        #print('Testing...')
        with torch.no_grad():
            x = data.float().to(self.device)
            z = self.net(x)
            score = torch.sum((z - self.c) ** 2, dim=1)[0]

        #print('score: {}'.format(score.detach().cpu().numpy()))
        return score.detach().cpu().numpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--test_data_dir', type=str, default='../datasets/test/melspectrograms')
    parser.add_argument('-prm', '--parameters_path', type=str, default='weights/network_parameters.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--tsne', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)

    args = parser.parse_args()

    test_data = get_mydata(args, mode='test')
    Network = TesterDeepSVDD(args.parameters_path, args.latent_dim)

    Network.test_dataset(test_data, args.tsne)
    #Network.test_dataset_PN(test_data, 0.696)
