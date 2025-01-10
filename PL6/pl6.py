# Implement Digit recognition from scratch 
# one layer neural network
# backpropagation

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def load_data_X(file):
    df = read_csv(file)
    X = df.values
    return X 

def load_data_T(file):
    df = read_csv(file)
    T = df.values
    return T

def image_visualization(X):
    dig = X[:,36] # dimension (25,1)
    plt.figure(1)
    plt.imshow(dig.reshape(5,5), cmap='gray')
    plt.show()
    
# Função de ativação sigmoidal
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmoidal
def sigmoid_derivative(x):
    return x * (1 - x)

# Inicialização de pesos e bias
def initialize_weights(nx, nh1, ny, n):
    W1 = np.random.random((nh1, nx)) * 0.01  # Pesos da entrada para camada oculta
    B1 = np.random.random((nh1, 1)) * 0.01  # Bias para a camada oculta
    W2 = np.random.random((ny, nh1)) * 0.01  # Pesos da camada oculta para saída
    B2 = np.random.random((ny, 1)) * 0.01  # Bias para a camada de saída
    MI = np.ones((n, 1))  # Matriz de uns para o cálculo do bias
    return W1, B1, W2, B2, MI

# Passagem forward
def forward_pass(X, W1, B1, W2, B2):
    a0 = X  # Entrada
    a1 = np.dot(W1, a0) + B1  # Soma ponderada na camada oculta
    y1 = sigmoid(a1)  # Ativação sigmoidal
    a2 = np.dot(W2, y1) + B2  # Soma ponderada na saída
    y2 = sigmoid(a2)  # Ativação sigmoidal na saída
    return a0, a1, y1, a2, y2

# Cálculo do erro
def compute_error(T, y2):
    e2 = T - y2  # Erro na saída
    erro = np.mean(np.square(e2))  # Erro médio quadrático
    return e2, erro

# Backpropagation
def backpropagation(W1, B1, W2, B2, a1, y1,y2, a0, e2, alfa, MI):
    S2 = e2 * sigmoid_derivative(y2)  # Gradiente da camada de saída
    dW2 = alfa * np.dot(S2, a1.T)  # Gradiente dos pesos W2
    dB2 = alfa * np.dot(S2, MI)  # Gradiente dos bias B2

    W2 += dW2
    B2 += dB2

    S1 = np.dot(W2.T, S2) * sigmoid_derivative(y1)  # Gradiente da camada oculta
    dW1 = alfa * np.dot(S1, a0.T)  # Gradiente dos pesos W1
    dB1 = alfa * np.dot(S1, MI)  # Gradiente dos bias B1

    W1 += dW1
    B1 += dB1

    return W1, B1, W2, B2

# Treinamento da rede
def train_network(X, T, epochs, alfa, W1, B1, W2, B2, MI):
    for numEP in range(epochs):
        a0, a1, y1, a2, y2 = forward_pass(X, W1, B1, W2, B2)  # Passagem forward
        e2, erro = compute_error(T, y2)  # Cálculo do erro
        W1, B1, W2, B2 = backpropagation(W1, B1, W2, B2, a1, y1, a0, e2, alfa, MI)  # Backpropagation

        # Exibir erro a cada 100 épocas
        if numEP % 100 == 0:
            print(f"Epoch {numEP}, Error: {erro}")

    return W1, B1, W2, B2


if __name__ == '__main__':
    # Carregar dados
    X = read_csv("digitsX.csv").values.T
    T_raw = read_csv("digitsT.csv").values.T

    # Normalizar dados e converter T para one-hot encoding
    X = X / 255.0  # Normalizar para o intervalo [0, 1]
    T = np.eye(10)[T_raw.flatten()].T  # One-hot encoding

    # Inicializar pesos
    nx = X.shape[0]  # Número de features (25)
    nh1 = 50  # Número de neurónios na camada oculta
    ny = T.shape[0]  # Número de classes (10)
    n = X.shape[1]  # Número de exemplos
    W1, B1, W2, B2, MI = initialize_weights(nx, nh1, ny, n)

    # Visualizar um dígito
    image_visualization(X, 36)

    # Treinar a rede
    alfa = 0.1  # Taxa de aprendizado
    epochs = 1000
    W1, B1, W2, B2 = train_network(X, T, epochs, alfa, W1, B1, W2, B2, MI)




if __name__ == '__main__':
    X = load_data_X("digitsX.csv")
    T = load_data_T("digitsT.csv")
    N = X.shape[1]
    image_visualization(X)