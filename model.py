import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

class Encoder(nn.Module):
    def __init__(self, z_dim=64):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, 3, stride=2, padding=1, bias=False)
        self.bn2d1 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False)
        self.bn2d2 = nn.BatchNorm2d(128, eps=1e-4, affine=False)
        self.conv3 = nn.Conv2d(128, 64, 3, stride=2, padding=1, bias=False)
        self.bn2d3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(64, 32, 3, stride=(2,1), padding=1, bias=False)
        self.bn2d4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 8 * 4, z_dim)
        #self.fc1 = nn.Linear(32 * 8 * 4, 256)
        #self.fc2 = nn.Linear(256, z_dim)

    def forward(self, x):
        x = F.relu(self.bn2d1(self.conv1(x)))
        x = F.relu(self.bn2d2(self.conv2(x)))
        x = F.relu(self.bn2d3(self.conv3(x)))
        x = F.relu(self.bn2d4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x

class Decoder(nn.Module):
    def __init__(self, z_dim=64):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, 32 * 8 * 4)
        #self.fc2 = nn.Linear(256, 32 * 8 * 4)
        self.deconv1 = nn.ConvTranspose2d(32, 64, (2,1), stride=(2,1), padding=0, bias=False)
        self.bn2d6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(64, 128, 2, stride=2, padding=0, bias=False)
        self.bn2d7 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(128, 256, 2, stride=2, padding=0, bias=False)
        self.bn2d8 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(256, 1, 2, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), 32, 8, 4)
        x = F.relu(self.bn2d6(self.deconv1(x)))
        x = F.relu(self.bn2d7(self.deconv2(x)))
        x = F.relu(self.bn2d8(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self, z_dim=64):
        super(Autoencoder, self).__init__()
        
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__=='__main__':
    model = Autoencoder(z_dim=64)
    summary(model, input_size=(64,1,128,32), col_names=["input_size", "output_size"])