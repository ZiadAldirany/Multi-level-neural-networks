import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import losses_functions as lf
import input_data

class PINN(nn.Module):
    def __init__(self, base_input_dim, layers, fact,fourier_feat_input = False, fourier_feat_g = False):
        super(PINN, self).__init__()

        self.activation = nn.Tanh()

        # Number of Fourier frequencies to use
        fact = round(fact)

        # Fourier frequencies for encoding Dirichlet boundary conditions
        self.ks = 2**torch.arange(0, fact+1).unsqueeze(0)

        self.fourier_feat_input = fourier_feat_input
        self.fourier_feat_g = fourier_feat_g

        if fourier_feat_input == True and fourier_feat_g== True:
            self.adjusted_layers = [2*(fact+1)*base_input_dim] + layers + [fact+1]
        elif fourier_feat_input == True and fourier_feat_g== False:
            self.adjusted_layers = [2*(fact+1)*base_input_dim] + layers + [1]
        elif fourier_feat_input == False and fourier_feat_g== True:
            self.adjusted_layers = [1] + layers + [fact+1]
        else:
            self.adjusted_layers = [1] + layers + [1]


        # Define the neural network layers
        self.linears = nn.ModuleList([nn.Linear(self.adjusted_layers[i], self.adjusted_layers[i+1])
                                      for i in range(len(self.adjusted_layers)-1)])

    def forward(self, x):
        
        if self.fourier_feat_input:
            # Generate Fourier features for the input
            fourier_features_input = torch.cat((torch.sin(self.ks * torch.pi * x), torch.cos(self.ks * torch.pi * x)), 1)
            # Pass through the neural network layers
            z = fourier_features_input
        else:
            # Pass x through the NN without Fourier features mapping
            z = x

        # hidden layers
        for i in range(len(self.linears) - 1):
            z = self.linears[i](z)
            z = self.activation(z)
        
        z = self.linears[-1](z)

        # output function
        if self.fourier_feat_g:
            # Generate Fourier features for zero Dirichlet BC
            fourier_features_g = torch.sin(self.ks * torch.pi * x)
            z = z * fourier_features_g
            z = torch.mean(z, 1, keepdim=True)
        else:
            # Inforce BC without Fourier features
            z = z * x * (1 - x)

        return z

    def param_init(self):
        """
        Initialize network parameters (weights and biases).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.constant_(m.bias, 0.1)

class CombinedPINN(nn.Module):
    def __init__(self):
        super(CombinedPINN, self).__init__()
        self.multi_nets = nn.ModuleList()  # To hold all  networks
        self.epsilons = []  # To hold scaling factors for networks

    def add_network(self, new_net, epsilon):
        self.multi_nets.append(new_net)
        self.epsilons.append(epsilon)

    def remove_network(self, index):
        # Ensure the index is valid
        if index < 0 or index >= len(self.multi_nets):
            raise ValueError("Index out of range.")

        # Remove the correction network and epsilon
        del self.multi_nets[index]
        del self.epsilons[index]

    def forward(self, x):
        u = 0
        for correction, epsilon in zip(self.multi_nets, self.epsilons):
            u += epsilon * correction(x)
        return u

def train_PINN(model, x_data, f_data, x_test, f_test, optimizer_choice="Adam", 
               epochs=5000, learning_rate=0.01, print_interval=500):
    """
    Trains a Physics-Informed Neural Network (PINN) model.

    Parameters:
    - model: An instance of the PINN class.
    - x_data: training input data points.
    - x_test: testing input data points.
    - f_data: training right-hand side values of the PDE.
    - f_test: testing right-hand side values of the PDE.
    - optimizer_choice: Choice of optimizer. Can be "Adam" or "LBFGS". Default is "Adam".
    - epochs: Number of training epochs (default is 5000).
    - learning_rate: Learning rate for the optimizer (default is 0.01).
    - print_interval: Number of epochs between printing progress (default is 500).

    """
        
    if optimizer_choice == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate,
                                      line_search_fn = "strong_wolfe", max_iter=20)
    else:
        raise ValueError(f"Optimizer choice {optimizer_choice} not recognized!")

    loss_history_train = []
    loss_history_test = []
    
    def closure():
            optimizer.zero_grad()
            u_pred = model(x_data)
            
            loss_global, _ = lf.poisson_pde_loss(u_pred, x_data, f_data)
                
            loss_global.backward(retain_graph = False)
            return loss_global

    # Training loop
    for epoch in range(epochs):

        if optimizer_choice == "LBFGS":
            optimizer.step(closure)
        
        else:
            closure()
            optimizer.step()

        # Print and store the loss and errors at intervals
        if (epoch+1) % print_interval == 0 or epoch == 0:
            u_pred = model(x_data)
            loss_train, _ = lf.poisson_pde_loss(u_pred, x_data, f_data)  
                
            u_test = model(x_test)
            loss_test, _ = lf.poisson_pde_loss(u_test, x_test, f_test)
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss train: {loss_train.item():.4e}, Loss test: {loss_test.item():.4e}')
            
            loss_history_train.append(loss_train.item())
            loss_history_test.append(loss_test.item())
            
    return np.array(loss_history_train), np.array(loss_history_test)


def train_multi_level(hp):
    # Generate the required data
    x_data = input_data.generate_data(domain_limits=[hp['x_left'],hp['x_right']], num_points=hp['N_points_train'][0])
    f_data = input_data.loss_RHS(x_data, hp['k'])

    x_test = input_data.generate_random_data(domain_limits=[hp['x_left'],hp['x_right']], num_points=hp['N_points_test'][0])
    f_test = input_data.loss_RHS(x_test, hp['k'])


    # Initial network to approximate u_0
    initial_net = PINN(base_input_dim=hp['dim'], layers=[hp['width'][0]], fact=hp['M'][0],
                       fourier_feat_input = hp['fourier_feat_input'], fourier_feat_g = hp['fourier_feat_g'])

    # Initialize the model first level
    PINN.param_init(initial_net)

    # product of the mu_i up to the level i
    mu_tot = hp['mu'][0]

    print(f'\n Training of Level 0 \n')

    # Train the model using Adam and LBFGS optimizer
    if hp['Adam'][0]:
        loss_train, loss_test = train_PINN(initial_net, x_data, f_data, x_test, f_test,
                            optimizer_choice="Adam", epochs=hp['niter_adam'][0],
                            learning_rate=hp['lr_adam'][0], print_interval=hp['print_interval_adam'])
        print("Adam : Done !")

    if hp['LBFGS'][0]:
        loss_train, loss_test = train_PINN(initial_net, x_data, f_data, x_test, f_test,
                            optimizer_choice="LBFGS", epochs=hp['niter_bfgs'][0],
                            learning_rate=hp['lr_bfgs'][0], print_interval= hp['print_interval_bfgs'])
        print("LBFGS : Done !")

    # Create the combined model with the initial network
    combined_model = CombinedPINN()
    combined_model.add_network(initial_net, epsilon=hp['mu'][0])

    # Correction networks loop
    for j in range (1, hp['lvls']):
        # Normalization factor
        mu_i = hp['mu'][j]
        mu_tot = mu_tot*mu_i

        # Calculate the scaled RHS term of the correction using the residual of the previous approx
        u_pred_data = combined_model.multi_nets[j-1](x_data)
        u_pred_test = combined_model.multi_nets[j-1](x_test)
        _ , R_data = lf.poisson_pde_loss(u_pred_data, x_data, f_data)
        f_data = R_data*mu_i
        _ , R_test = lf.poisson_pde_loss(u_pred_test, x_test, f_test)
        f_test = R_test*mu_i

        # Create the correction network
        correction_net = PINN(base_input_dim=hp['dim'], layers=[hp['width'][j]], fact=hp['M'][j],
                       fourier_feat_input = hp['fourier_feat_input'], fourier_feat_g = hp['fourier_feat_g'])
        PINN.param_init(correction_net)

        # Add the correction network to the combined model with the normalization factor
        combined_model.add_network(correction_net, epsilon=1/mu_tot)

        print(f'\n Training of Level {j} \n')

        # Train the model using Adam and LBFGS optimizer
        if hp['Adam'][j]:
            loss_train, loss_test = train_PINN(correction_net, x_data, f_data, x_test, f_test,
                            optimizer_choice="Adam", epochs=hp['niter_adam'][j],
                            learning_rate=hp['lr_adam'][j], print_interval=hp['print_interval_adam'])
            print("Adam : Done !")

        if hp['LBFGS'][j]:
            loss_train, loss_test = train_PINN(correction_net, x_data, f_data, x_test, f_test,
                           optimizer_choice="LBFGS", epochs=hp['niter_bfgs'][j],
                           learning_rate=hp['lr_bfgs'][j], print_interval=hp['print_interval_bfgs'])
            print("LBFGS : Done !")

    return combined_model