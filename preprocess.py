import torch
import numpy as np
import os
import sys
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

class MyData_loader(data.Dataset):
    def __init__(self, data, label=None, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            out_data = self.transform(self.data).permute(1,2,0)[idx].unsqueeze(0)

        if self.label is not None:
            out_label = torch.tensor(self.label[idx])
            return out_data, out_label

        return out_data


def load_dataset_for_train(spectrograms_path):
    data = []
    label = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if os.path.split(root)[1] == 'miss':
                label.append(-1)
            elif os.path.split(root)[1] == 'normal':
                label.append(1)
            else:
                label.append(0)
            spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            data.append(spectrogram)
            #print('{} is OK.'.format(file_name))
    data = np.array(data)

    return data, label

def load_dataset_for_validation(spectrograms_path):
    data = []
    label = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)

            if os.path.split(root)[1] == 'normal':
                label.append(0)
            else:
                label.append(1)
            
            spectrogram = np.load(file_path)
            data.append(spectrogram)

    data = np.array(data)

    return data, label

def load_dataset_for_pretrain(spectrograms_path):
    data = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if os.path.split(root)[1] == 'miss':
                continue
            else:
                spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
                data.append(spectrogram)

    data = np.array(data)

    return data

def load_dataset_for_pretrain_validation(spectrograms_path):
    data = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if os.path.split(root)[1] == 'normal':
                spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
                data.append(spectrogram)
            else:
                continue

    data = np.array(data)

    return data

def load_dataset_for_test(spectrograms_path):
    data = []
    label = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)

            if os.path.split(root)[1] == 'normal':
                label.append(0)
            elif os.path.split(root)[1] == 'buzz':
                label.append(1)
            elif os.path.split(root)[1] == 'mid-buzz':
                label.append(2)
            elif os.path.split(root)[1] == 'muffled':
                label.append(3)
            elif os.path.split(root)[1] == 'mute':
                label.append(4)
            elif os.path.split(root)[1] == 'others':
                label.append(5)
            elif os.path.split(root)[1] == 'tmp':
                label.append(6)  
            else:
                print("The name of the directory must be set to 'Miss' or 'Normal'.")
                sys.exit()

            spectrogram = np.load(file_path)
            data.append(spectrogram)

    data = np.array(data)

    return data, label

def get_mydata(args, mode):

    transform = transforms.Compose([transforms.ToTensor()])

    if mode == 'train':
        train_data, labels = load_dataset_for_train(args.data_dir)
        train_dataset = MyData_loader(train_data, label=labels, transform=transform)
        data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    elif mode == 'pretrain':
        pretrain_data = load_dataset_for_pretrain(args.data_dir)
        pretrain_dataset = MyData_loader(pretrain_data, transform=transform)
        data_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    elif mode == 'val':
        val_data, labels = load_dataset_for_validation(args.val_data_dir)
        val_dataset = MyData_loader(val_data, label=labels, transform=transform)
        data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False) 
    elif mode == 'pretrain_val':
        pretrain_val_data = load_dataset_for_pretrain_validation(args.val_data_dir)
        pretrain_val_dataset = MyData_loader(pretrain_val_data, transform=transform)
        data_loader = DataLoader(pretrain_val_dataset, batch_size=args.batch_size,
                                  shuffle=False)
    elif mode == 'test':
        test_data, labels = load_dataset_for_test(args.test_data_dir)
        test_dataset = MyData_loader(test_data, label=labels, transform=transform)
        data_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False)
        

    return data_loader