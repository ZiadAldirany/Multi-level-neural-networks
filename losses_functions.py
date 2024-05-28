import torch

def poisson_pde_loss(u, x,  f):
    u_grad = torch.autograd.grad(outputs=u, inputs=x,
                             grad_outputs=torch.ones_like(u),
                             create_graph=True, retain_graph=True)[0]
    u_laplacian = torch.autograd.grad(outputs=u_grad, inputs=x,
                                  grad_outputs=torch.ones_like(u_grad),
                                  create_graph=True, retain_graph=True)[0]
    R = u_laplacian + f
    loss = torch.mean(torch.square(R))

    return loss, R.detach()
