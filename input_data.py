import torch
import numpy as np

def generate_data(domain_limits, num_points):
    # Generate points equally spaced
    x = torch.linspace(domain_limits[0],domain_limits[1],num_points)

    return x.unsqueeze(-1).requires_grad_(True)

def generate_random_data(domain_limits, num_points):

    # Generate points randomly between x_min and x_man

    x = torch.rand(num_points)
    x = x * (domain_limits[1] - domain_limits[0]) + domain_limits[0]

    return x.unsqueeze(-1).requires_grad_(True)

def loss_RHS(x, k):
    # RHS of Poisson problem
    f = - exact_ddu(x, k)
    return f.detach()

def exact_u(x, k):
    # Exact solution of Poisson problem
    u = (torch.exp(torch.sin(k*torch.pi*x))-1 + (x**3-x))
    return u.detach()

def exact_du(x, k):
    # Exact derivative of Poisson problem
    du = (k*torch.pi*torch.cos(k*torch.pi*x)*torch.exp(torch.sin(k*torch.pi*x)) + (3*x**2-1))
    return du.detach()

def exact_ddu(x, k):
    # Exact second derivative of Poisson problem
    ddu = (k*torch.pi)**2*(torch.cos(k*torch.pi*x)**2 - torch.sin(k*torch.pi*x))*torch.exp(torch.sin(k*torch.pi*x)) + 6*x
    return ddu.detach()



