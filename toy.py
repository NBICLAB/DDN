import torch
import math
from torch.distributions import Uniform, Bernoulli, Normal, Uniform

class Toy2D_Task1():
    # Squares
    def sample(self, num_samples):
        x = Uniform(-1, 1).sample((num_samples, 1))       
        a = Bernoulli(0.5).sample((num_samples, 1))        
        b = Uniform(-5 + x, -1 + x).sample()           
        c = Uniform(1 - x, 5 - x).sample()             
        y0 = a * b + (1 - a) * c                       
        b = Uniform(-5 + x, -1 + x).sample()           
        c = Uniform(1 - x, 5 - x).sample()             
        y1 = a * b + (1 - a) * c    
        y = torch.cat([y0, y1], dim=1) 
        return x, y
    def ygivenx_pdf(self, x, y):
        density = torch.zeros_like(x)
        mask0 = (y[:, :1] > (-5 + x)) & (y[:, :1] < (-1 + x)) & (y[:, 1:] > (-5 + x)) & (y[:, 1:] < (-1 + x))
        mask1 = (y[:, :1] > (1 - x)) & (y[:, :1] < (5 - x)) & (y[:, 1:] > (1 - x)) & (y[:, 1:] < (5 - x))  
        density[mask0 | mask1] = 1 / 32
        return density

class Toy2D_Task2():
    # Half Gaussian
    def sample(self, num_samples):
        x = Uniform(-1, 1).sample((num_samples, 1))
        r = torch.cat([ torch.cos(x * math.pi), 
            -torch.sin(x * math.pi), 
            torch.sin(x * math.pi), torch.cos(x * math.pi)], dim=1).reshape(-1, 2, 2)
        y0 = Normal(0, 2).sample((num_samples, 1))
        y1 = Normal(0, 2).sample((num_samples, 1))
        y0 = torch.abs(y0)
        y = torch.cat([y0, y1], dim=1)
        y = torch.bmm(r, y.unsqueeze(-1)).squeeze(-1)
        return x, y
    def ygivenx_pdf(self, x, y):
        density = torch.zeros_like(x)
        A = torch.cat([
            torch.cos(x * math.pi), 
            torch.sin(x * math.pi), 
            -torch.sin(x * math.pi), 
            torch.cos(x * math.pi)], 
        dim=1).reshape(-1, 2, 2)
        y = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
        mask = y[:, 0] > 0
        density = Normal(torch.zeros_like(y), 2 * torch.ones_like(y)).log_prob(y).sum(dim=1).exp()
        density[~mask] = 0
        density[mask] = 2 * density[mask]
        return density

class Toy2D_Task3():
    # Gaussian Stick
    def sample(self, num_samples):
        x = Uniform(-1, 1).sample((num_samples, 1))
        c = (-0.75 + x) / 2
        r = torch.cat([ torch.cos(c * math.pi), 
            -torch.sin(c * math.pi), 
            torch.sin(c * math.pi), torch.cos(c * math.pi)], dim=1).reshape(-1, 2, 2)
        y0 = Normal(0, 1).sample((num_samples, 1))
        y1 = Uniform(-6, 6).sample((num_samples, 1))
        y = torch.cat([y0, y1], dim=1)
        y = torch.bmm(r, y.unsqueeze(-1)).squeeze(-1)
        return x, y

    def ygivenx_pdf(self, x, y):
        density = torch.zeros_like(x)
        c = (-0.75 + x) / 2
        A = torch.cat([
            torch.cos(c * math.pi), 
            torch.sin(c * math.pi), 
            -torch.sin(c * math.pi), 
            torch.cos(c * math.pi)], 
        dim=1).reshape(-1, 2, 2)
        
        y = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)

        density = (Normal(torch.zeros_like(y[:, 0]), torch.ones_like(y[:, 0])).log_prob(y[:, 0]) +\
            Uniform(-6 * torch.ones_like(y[:, 1]), 6 * torch.ones_like(y[:, 1]), 
            validate_args=False).log_prob(y[:, 1])).exp()
        return density

class Toy2D_Task4():
    # Elastic Ring
    def sample(self, num_samples):
        x = Uniform(-1, 1).sample((num_samples, 1))
        d = Uniform(0, 2).sample((num_samples, 1))
        theta = Uniform(0, 2 * math.pi).sample((num_samples, 1))
        y0 = (4 + 2 * x + d) * torch.cos(theta)
        y1 = (4 - 2 * x + d) * torch.sin(theta)
        y = torch.cat([y0, y1], dim=1)
        return x, y

    def ygivenx_pdf(self, x, y):
        density = torch.zeros_like(x)
        mask0 = (
            y[:, :1] / (4 + 2 * x)          
        ) ** 2 + (
            y[:, 1:] / (4 - 2 * x)
        ) ** 2 > 1  
        mask1 = (
            y[:, :1] / (6 + 2 * x)    
        ) ** 2 + (
            y[:, 1:] / (6 - 2 * x)
        ) ** 2 < 1 
        area = math.pi * ((6 + 2 * x)  * (6 - 2 * x) - (4 + 2 * x)  * (4 - 2 * x) )
        density[mask0 & mask1] += 1 / area[mask0 & mask1]
        return density
