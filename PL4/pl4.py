#Paro de parar uma rede neural, quando o erro quadrático médio entre a saída da rede e a saída desejada for menor que um valor de tolerância.
#Treinar uma rede neuronal é um processo iterativo que envolve ajustar os pesos sinápticos da rede para minimizar o erro entre a saída da rede e a saída desejada.
#Obrigatório que as funções sejam deriváveis, para aplicar o gradiente.

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np

#train a perceptron network
#Perceptron Training rule (on-line)
#1. Initialize the weights and bias to small random values
#2. For each input vector x:
#   a. Compute the output of the perceptron
#   b. Update the weights and bias using the perceptron training rule
#3. Repeat steps 2-4 until the error is below a certain threshold or a maximum number of iterations is reached

def load_data_withoutHeader(nome_ficheiro):
    df = pd.read_csv(nome_ficheiro, header=None)
    return df

def load_data(nome_ficheiro):
    df = pd.read_csv(nome_ficheiro)
    return df

def perceptronTrainingRule(input, output, learning_rate, max_iterations):
    # Initialize the weights and bias to small random values
    weights = np.random.rand(input.shape[1])
    bias = np.random.rand()
    
    for _ in range(max_iterations):
        total_error = 0
        for i in range(input.shape[0]):
            # Compute the output of the perceptron
            y = np.dot(input[i], weights) + bias
            # Activation function: hardlim
            y = 1 if y >= 0 else 0
            # Calculate the error
            error = output[i] - y
            # Update the weights and bias using the perceptron training rule
            weights += learning_rate * error * input[i]
            bias += learning_rate * error
            total_error += abs(error)
        # Stop if the total error is below the threshold
        if total_error == 0.01:
            break
        
    return weights, bias

def plot_decision_boundary(input, output, weights, bias):
    # Plot the data points
    plt.scatter(input[:, 0], input[:, 1], c=output, cmap=ListedColormap(['red', 'blue']))
    # Plot the decision boundary
    x = np.linspace(input[:, 0].min(), input[:, 0].max(), 100)
    y = -(weights[0] * x + bias) / weights[1]
    plt.plot(x, y, color='green')
    plt.show()
    


if __name__ == "__main__":
    df_linear = load_data('PL4/P4_data1.csv')
    df_nonlinear = load_data_withoutHeader('PL4/P4_data2.csv')
    weights_linear, bias_linear = perceptronTrainingRule(df_linear[['x1', 'x2']].to_numpy(), df_linear['T'].to_numpy(), 0.1, 100)
    plot_decision_boundary(df_linear[['x1', 'x2']].to_numpy(), df_linear['T'].to_numpy(), weights_linear, bias_linear)
    weights_nonlinear, bias_nonlinear = perceptronTrainingRule(df_nonlinear[[0, 1]].to_numpy(), df_nonlinear[2].to_numpy(), 0.1, 100)
    plot_decision_boundary(df_nonlinear[[0, 1]].to_numpy(), df_nonlinear[2].to_numpy(), weights_nonlinear, bias_nonlinear)
    