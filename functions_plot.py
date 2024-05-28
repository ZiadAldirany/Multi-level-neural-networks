import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import input_data

        
def main_plot(combined_model, hp):

    # data plot
    x_data = input_data.generate_data(domain_limits=[hp['x_left'],hp['x_right']], num_points=hp['N_points_plot'][0])
    x_plt = x_data.detach().cpu().numpy()
    u_plt= combined_model(x_data).detach().cpu().numpy()
    u_ex_plt = input_data.exact_u(x_data, hp['k']).detach().cpu().numpy()

    # plot approximated solution
    plt.figure(figsize=(6,3))
    u_pred_data = combined_model(x_data)
    plt.subplot(1,2,1)
    plt.plot(x_plt, u_plt, '.')
    plt.xlabel('$x$')
    plt.ylabel('$\~{u}$')

    # plot the error 
    plt.subplot(1,2,2)
    plt.plot(x_plt, u_ex_plt - u_plt, '.')
    plt.xlabel('$x$')
    plt.ylabel('$u_{ex} - \~{u}$')

    # plot u_i at each level
    plt.figure(figsize=(20,10//(hp['lvls']+1)))
    for i in range(0,hp['lvls']):
        plt.subplot(1,hp['lvls']+1,i+1)

        u_plt = combined_model.multi_nets[i](x_data).detach().cpu().numpy()
        plt.plot(x_plt,u_plt,'.')
        plt.title('u'+str(i))
        plt.xlabel('x')
        plt.ylabel('u')

    
    
