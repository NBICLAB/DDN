import os
import torch
import urllib
import pandas as pd
import zipfile
import numpy as np

from torch.utils import data

class UCIDataset():
    def __init__(self, name, data_path='data'):
        self.datasets = {
            'concrete': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
            'fish': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv',
            'energy': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx',
            'parkinsons': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data',
            'temperature': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00514/Bias_correction_ucl.csv',
            'air' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip',
            'skillcraft': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv'
        }
        self.data_path = data_path
        self.name = name
        self._load_dataset()
    
    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception('Unknown dataset!')
        if not os.path.exists(os.path.join(os.getcwd(), self.data_path, 'UCI')):
            os.makedirs(os.path.join(os.getcwd(), self.data_path, 'UCI'))

        self.x_dim = None
        self.data = None
        self.targets = None
        self.y_dim = None
        url = self.datasets[self.name]
        file = os.path.join(self.data_path, 'UCI', url.split('/')[-1])
        if not os.path.exists(file):
            urllib.request.urlretrieve(url, file)
        if self.name == 'concrete':
            data = pd.read_excel(file, header=0).values
        elif self.name == 'energy':
            data = pd.read_excel(file, header=0, engine='openpyxl').values[:768, :10]
            self.x_dim = data.shape[1] - 2
            self.y_dim = 2
        elif self.name == 'air':
            dim = 3
            zip_dir = os.path.join(self.data_path, 'UCI', 'air')
            if not os.path.exists(zip_dir):
                zipfile.ZipFile(file).extractall(os.path.join(self.data_path, 'UCI', 'air'))
            file = os.path.join(self.data_path, 'UCI', 'air', 'AirQualityUCI.xlsx')
            data = pd.read_excel(file, header=0, engine='openpyxl').values[:, 2:-2].astype(np.float32)
            data[:, -dim:][data[:, -dim:] == -200] = np.nan 
            data = data[~np.isnan(data).any(axis=1)]
            data = torch.from_numpy(data)
            self.data = data[:, :-dim]
            self.targets = data[:, -dim:]
            self.x_dim = data.shape[1] - dim
            self.y_dim = dim
        elif self.name == 'fish':
            data = pd.read_csv(file, header=None, delimiter=';').values
        elif self.name == 'parkinsons':
            data = pd.read_csv(file, header=0, delimiter=',').values[:, 4:]
            data = torch.from_numpy(data).float()
            self.data = data[:, 2:]
            self.targets = data[:, :2]
            self.x_dim = data.shape[1] - 2
            self.y_dim = 2
        elif self.name == 'temperature':
            data = pd.read_csv(file, header=0, delimiter=',')
            data = data.replace('NaN', np.nan).dropna().values[:, 2:].astype(np.float32)
            data = torch.from_numpy(data)
            self.data = data[:, :-2]
            self.targets = data[:, -2:]
            self.x_dim = data.shape[1] - 2
            self.y_dim = 2          
        elif self.name == 'skillcraft':
            data = pd.read_csv(file, header=0, delimiter=',').replace(
                '?', np.nan).dropna().values[:, 1:].astype(np.float32)
            data = torch.from_numpy(data)
            self.targets = data[:, 10:14]
            self.data = torch.cat([data[:, 0:10], data[:, 14:]], dim=1) 
            self.x_dim = self.data.shape[1]
            self.y_dim = self.targets.shape[1]
        if self.x_dim == None:
            self.x_dim = data.shape[1] - 1
        if self.y_dim == None:
            self.y_dim = 1        
        if self.data == None:
            self.data = torch.from_numpy(data[:, :self.x_dim]).float()
        if self.targets == None:
            self.targets = torch.from_numpy(data[:, -self.y_dim:]).float()
