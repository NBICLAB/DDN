import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from base import BaseMLPWithEqualMapping, VariationalLayer, BaseMLP
from utils import perm_generator

class DeconvEstimator(nn.Module):
    def __init__(self, in_channels, out_channels_list, groups, scale_factor):
        super(DeconvEstimator, self).__init__()

        channels = [in_channels,] + out_channels_list
        self.layers = []
        for i in range(len(channels) - 1):
            self.layers.extend([
                nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False),
                nn.Conv1d(channels[i] * groups, channels[i + 1] * groups, 
                    kernel_size=3, stride=1, padding=1, groups=groups),
                nn.BatchNorm1d(channels[i + 1] * groups),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        self.layers.extend([
                nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False),
                nn.Conv1d(channels[-1] * groups, 1 * groups, kernel_size=3, stride=1, padding=1, groups=groups),
        ])  
        self.layers = nn.Sequential(*self.layers)  
    def forward(self, x):
        return self.layers(x)

class DDN(nn.Module):
    def __init__(self, 
        condition_dim, 
        target_dim, 
        target_range_list, 
        codes, 
        channels, 
        width, 
        beta,
        estimator_channels_list=[32, 16],
        scale_factor=4,
        num_perm=5,
        disable_vl=False,
        ):
        super(DDN, self).__init__()
        self.condition_dim = condition_dim
        self.target_dim = target_dim
        self.target_range_list = target_range_list
        self.channels = channels
        self.width = width
        self.beta = beta
        self.codes = codes
        self.num_perm = num_perm        
        self.num_bins = width * scale_factor ** (len(estimator_channels_list) + 1) 
        self.disable_vl = disable_vl

        self.unary = False

        if self.target_dim == 1: 
            self.encoder = BaseMLP(self.condition_dim)
        else:
            self.encoder = BaseMLPWithEqualMapping(self.condition_dim, 2 * self.target_dim)
        
        if self.disable_vl:
            self.latent_layer = nn.Sequential(
                nn.Linear(self.encoder.out_dim, codes),
                nn.LeakyReLU(0.2)
            )
        else:
            self.latent_layer = VariationalLayer(self.encoder.out_dim, codes)

        self.conv_linears = nn.Sequential(
            nn.Conv1d(
                in_channels=target_dim * codes, 
                out_channels=target_dim * width * channels, 
                kernel_size=1, groups=target_dim),
            nn.LeakyReLU(0.2)
        )
        self.estimator = DeconvEstimator(channels, estimator_channels_list, target_dim, scale_factor)
        self.bins = torch.stack([torch.linspace(target_range[0], target_range[1], self.num_bins + 1) 
            for target_range in target_range_list])
        self.target_range = torch.tensor(self.target_range_list, dtype=torch.float32) 

        max_perm_num = math.factorial(self.target_dim)
        if self.num_perm > max_perm_num:
            self.num_perm = max_perm_num

        if self.target_dim > 1:
            self.__generate_perm()
            self.__generate_mask_set()

    def __generate_perm(self):
        perm_gen = perm_generator(self.target_dim)
        perms = [next(perm_gen) for i in range(self.num_perm)]
        self.perms = torch.tensor(perms)

    def __generate_mask_set(self):
        self.masks = torch.zeros(self.num_perm * self.target_dim, self.target_dim)
        for i in range(self.num_perm):
            for j in range(self.target_dim):
                self.masks[i * self.target_dim + j, self.perms[i, :j]] = 1
        self.masks = torch.unique(self.masks, dim=0)

    def forward(self, input, targets, mask=None):        
        if self.target_dim > 1:        
            if mask is None:
                rnd_mask_idx = torch.randint(0, self.masks.shape[0], (targets.shape[0],))
                mask = self.masks[rnd_mask_idx]

            mask = mask.to(targets.device)
            masked_targets = targets * mask        
    
            x = self.encoder(input, torch.cat([masked_targets, mask], dim=1))
        else:
            x = self.encoder(input)
        if self.disable_vl:
            z = self.latent_layer(x)
            kld = 0
        else:
            z, kld = self.latent_layer(x) 
        zs = z.repeat(1, self.target_dim).unsqueeze(-1)
        y = self.conv_linears(zs).view(-1, self.target_dim * self.channels, self.width)
        logits = self.estimator(y)

        log_mess = F.log_softmax(logits, dim=2)

        if self.training:
            targets_bin_idx = torch.searchsorted(self.bins, targets.T.contiguous(), right=True).T - 1
            loss = F.nll_loss(log_mess.reshape(-1, self.num_bins), targets_bin_idx.flatten())
            loss += self.beta * kld        
            return log_mess, loss
        else:
            return log_mess

    def loss(self, outputs, targets):
        log_mess, loss = outputs
        return loss

    @torch.no_grad()
    def __conditional_bin_logprobs(self, x, y, mask=None):
        self.eval()
        self.target_range = self.target_range.to(y.device)
        bin_logmess = self.forward(x, y, mask)
        bin_logprobs = bin_logmess - torch.log(self.target_range[:, 1:] - self.target_range[:, :1])\
             + math.log(self.num_bins) 
        return bin_logprobs

    def multi_path_conditional_bin_logprobs(self, x, y):
        
        masks = torch.zeros(self.num_perm, self.target_dim, *y.shape, device=y.device)
        for i in range(self.num_perm):
                for j in range(self.target_dim):
                    masks[i, j, :, self.perms[i, :j]] = 1

        x = x.expand(self.num_perm, self.target_dim, -1, -1).reshape(-1, x.shape[1])
        y = y.expand_as(masks).reshape(-1, y.shape[1])
        masks = masks.view_as(y)
        bin_logprobs = self.__conditional_bin_logprobs(x, y, masks)
        bin_logprobs = bin_logprobs.view(self.num_perm * self.target_dim, -1, self.target_dim, self.num_bins)
        bin_logprobs = bin_logprobs[torch.arange(self.num_perm * self.target_dim), :, 
            self.perms.flatten(), :].view(self.num_perm, self.target_dim, -1, self.num_bins)

        for i in range(self.num_perm):        
            idx = (self.perms[i] == torch.arange(self.target_dim).unsqueeze(1)).nonzero(as_tuple=True)[1]
            bin_logprobs[i, torch.arange(self.target_dim), :, :] = bin_logprobs[i, idx, :, :]
        
        bin_logprobs = bin_logprobs.permute(2, 0, 1, 3)
        return bin_logprobs
    
    def __logprobs_from_bins(self, bin_logprobs, y):
        bin_logprobs = F.pad(bin_logprobs, (1, 1, 0, 0), value=-math.inf) 
        y_bin_idx = torch.searchsorted(self.bins, y.T.contiguous(), right=True).T
        y_bin_idx = y_bin_idx[:, None, :, None].expand(-1, self.num_perm, -1, -1)
        log_probs = torch.gather(bin_logprobs, -1, y_bin_idx)
        log_probs = torch.logsumexp(log_probs.sum(dim=-2), dim=1, keepdim=True) - math.log(self.num_perm)
        return log_probs

    def conditional_logprobs(self, x, y, batch_size=64):
        if self.target_dim > 1:
            log_probs_list = []            
            Xs = torch.split(x, batch_size, dim=0)
            Ys = torch.split(y, batch_size, dim=0)
            for i, (x, y) in enumerate(zip(Xs, Ys)):
                bin_logprobs = self.multi_path_conditional_bin_logprobs(x, y)
                log_probs = self.__logprobs_from_bins(bin_logprobs, y)
                log_probs_list.append(log_probs)
            log_probs = torch.cat(log_probs_list, dim=0)
            return log_probs
        else:
            bin_logprobs = self.__conditional_bin_logprobs(x, y).unsqueeze(1)
            log_probs = self.__logprobs_from_bins(bin_logprobs, y).squeeze(1)
        return log_probs

    def __sample_chain_rule(self, condition, perm):
        samples = torch.zeros(condition.shape[0], self.target_dim).to(condition.device)
        mask = torch.zeros_like(samples)
        for dim_idx in perm:
            with torch.no_grad():
                bin_logmess = self.forward(condition, samples, mask)
            dist = Categorical(probs=bin_logmess[:, dim_idx, :].exp())
            bin_idx = dist.sample()
            v = self.bins[dim_idx, bin_idx] 
            v = v + torch.rand_like(v) * (self.target_range[dim_idx, 1] - self.target_range[dim_idx, 0]) / self.num_bins
            mask[:, dim_idx] = 1
            samples[:, dim_idx] = v
        return samples

    def multi_path_sample(self, condition):
        samples = []
        for perm in self.perms:
            samples.append(self.__sample_chain_rule(condition, perm))
        samples = torch.stack(samples)
        idx = Categorical(logits=torch.ones(condition.shape[0], len(self.perms))).sample()
        return samples[idx, torch.arange(samples.shape[1])]
    
    def sample(self, condition):
        self.eval()
        if self.target_dim > 1:
            return self.multi_path_sample(condition)
        else:
            samples = torch.zeros(condition.shape[0], self.target_dim).to(condition.device)
            with torch.no_grad():
                bin_logmess = self.forward(condition, samples)
            dist = Categorical(probs=bin_logmess[:, 0, :].exp())
            bin_idx = dist.sample()
            v = self.bins[0, bin_idx] 
            v = v + torch.rand_like(v) * (self.target_range[0, 1] - self.target_range[0, 0]) / self.num_bins
            return v

    def _apply(self, fn):
        super(DDN, self)._apply(fn)
        self.bins = fn(self.bins)
        return self         
