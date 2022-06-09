import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

class network(nn.Module):
    def __init__(self, z_dim=64):
        super(network, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, 3, stride=2, padding=1, bias=False)
        self.bn2d1 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False)
        self.bn2d2 = nn.BatchNorm2d(128, eps=1e-4, affine=False)
        self.conv3 = nn.Conv2d(128, 64, 3, stride=2, padding=1, bias=False)
        self.bn2d3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(64, 32, 3, stride=(2,1), padding=1, bias=False)
        self.bn2d4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        #self.conv5 = nn.Conv2d(64, 32, 3, stride=(2,1), padding=1)
        #self.bn2d5 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 4, z_dim)
        #self.bn1d = nn.BatchNorm1d(self.z_dim, eps=1e-04, affine=False)

    def forward(self, x):
        x = F.relu(self.bn2d1(self.conv1(x)))
        x = F.relu(self.bn2d2(self.conv2(x)))
        x = F.relu(self.bn2d3(self.conv3(x)))
        x = F.relu(self.bn2d4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class autoencoder(nn.Module):
    def __init__(self, z_dim=64):
        super(autoencoder, self).__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(1, 256, 3, stride=2, padding=1, bias=False)
        self.bn2d1 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False)
        self.bn2d2 = nn.BatchNorm2d(128, eps=1e-4, affine=False)
        self.conv3 = nn.Conv2d(128, 64, 3, stride=2, padding=1, bias=False)
        self.bn2d3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(64, 32, 3, stride=(2,1), padding=1, bias=False)
        self.bn2d4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        #self.conv5 = nn.Conv2d(64, 32, 3, stride=(2,1), padding=1)
        #self.bn2d5 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 4, self.z_dim)
        #self.bn1d = nn.BatchNorm1d(self.z_dim, eps=1e-04, affine=False)

        self.fc2 = nn.Linear(self.z_dim, 32 * 8 * 4)
        self.deconv1 = nn.ConvTranspose2d(32, 64, (2,1), stride=(2,1), padding=0, bias=False)
        self.bn2d6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(64, 128, 2, stride=2, padding=0, bias=False)
        self.bn2d7 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(128, 256, 2, stride=2, padding=0, bias=False)
        self.bn2d8 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(256, 1, 2, stride=2, padding=0, bias=False)

    def encode(self, x):
        x = F.relu(self.bn2d1(self.conv1(x)))
        x = F.relu(self.bn2d2(self.conv2(x)))
        x = F.relu(self.bn2d3(self.conv3(x)))
        x = F.relu(self.bn2d4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def decode(self, x):
        x = self.fc2(x)
        x = x.view(x.size(0), 32, 8, 4)
        x = F.relu(self.bn2d6(self.deconv1(x)))
        x = F.relu(self.bn2d7(self.deconv2(x)))
        x = F.leaky_relu(self.bn2d8(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

class network_(nn.Module):
    def __init__(self, z_dim=64):
        super(network, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, 3, padding=1, bias=False)
        self.bn2d1 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.bn2d2 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
        self.bn2d3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1, bias=False)
        self.bn2d4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 8 * 4, z_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn2d1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2d2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2d3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2d4(self.conv4(x)))
        x = F.max_pool2d(x, (2,1))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class autoencoder_(nn.Module):
    def __init__(self, z_dim=64):
        super(autoencoder, self).__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(1, 256, 3, padding=1, bias=False)
        self.bn2d1 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.bn2d2 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
        self.bn2d3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1, bias=False)
        self.bn2d4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 8 * 4, z_dim)

        #self.fc2 = nn.Linear(self.z_dim, 32 * 8 * 4)
        self.deconv1 = nn.ConvTranspose2d(int(self.z_dim / (8 * 4)), 32, 3, padding=1, bias=False)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(32, 64, 3, padding=1, bias=False)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d7 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 128, 3, padding=1, bias=False)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d8 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(128, 256, 3, padding=1, bias=False)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d9 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(256, 1, 3, padding=1, bias=False)

    def encode(self, x):
        x = F.leaky_relu(self.bn2d1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2d2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2d3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2d4(self.conv4(x)))
        x = F.max_pool2d(x, (2,1))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def decode(self, x):
        x = x.view(int(x.size(0)), int(self.z_dim / (8 * 4)), 8, 4)
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.bn2d6(self.deconv1(x)))
        x = F.interpolate(x, scale_factor=(2,1))
        x = F.leaky_relu(self.bn2d7(self.deconv2(x)))
        x = F.interpolate(x, scale_factor=2)
        x = F.leaky_relu(self.bn2d8(self.deconv3(x)))
        x = F.interpolate(x, scale_factor=2)
        x = F.leaky_relu(self.bn2d9(self.deconv4(x)))
        x = F.interpolate(x, scale_factor=2)
        x = torch.sigmoid(self.deconv5(x))
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

if __name__=='__main__':
    model = autoencoder(z_dim=64)
    summary(model, input_size=(64,1,128,32), col_names=["input_size", "output_size"])