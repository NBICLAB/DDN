import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim import Adam
from tqdm import tqdm

class VariationalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VariationalLayer, self).__init__()
        self.fc_mu = nn.Linear(in_features, out_features)
        self.fc_logvar = nn.Linear(in_features, out_features)
    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        if self.training:
            sigma = torch.exp(0.5 * logvar)
            z = mu + sigma * torch.randn_like(sigma)
        else:
            z = mu
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return z, kld.mean()

class BaseMLP(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
            nn.Linear(64, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.layers(x)


class BaseMLPWithEqualMapping(nn.Module):
    def __init__(self, x_dim, y_dim, out_dim=64):
        super().__init__()
        self.out_dim = out_dim

        self.x_mapping = nn.Sequential(
            nn.Linear(x_dim, 32),
        )

        self.y_mapping = nn.Sequential(
            nn.Linear(y_dim, 32),
        )
        self.bn = nn.BatchNorm1d(64)

        self.layers = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )
    def forward(self, x, y):
        x = self.x_mapping(x)
        y = self.y_mapping(y)
        inputs = torch.cat([x, y], dim=1)
        return self.layers(torch.tanh(self.bn(inputs)))


def train_one_epoch(net, optimizer, train_loader, device):
    net.train()
    runing_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        if net.unary:
            outputs = net(batch_x)
        else:
            outputs = net(batch_x, batch_y)
        loss = net.loss(outputs, batch_y)
        loss.backward()
        runing_loss += loss.item()
        optimizer.step()    
    return runing_loss / (step + 1)


def train(net, train_X, train_Y, device, num_epochs, batch_size, learning_rate=1e-3):
    net = net.to(device)
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(train_X, train_Y), 
        batch_size=batch_size, shuffle=True
    )    
    optimizer = Adam(net.parameters(), lr=learning_rate)
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_X, train_Y), 
        batch_size=batch_size, shuffle=True)
    for epoch in tqdm(range(num_epochs)):
        train_one_epoch(net, optimizer, train_loader, device)