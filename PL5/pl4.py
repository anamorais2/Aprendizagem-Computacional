#Paro de parar uma rede neural, quando o erro quadrático médio entre a saída da rede e a saída desejada for menor que um valor de tolerância.
#Treinar uma rede neuronal é um processo iterativo que envolve ajustar os pesos sinápticos da rede para minimizar o erro entre a saída da rede e a saída desejada.
#Obrigatório que as funções sejam deriváveis, para aplicar o gradiente.

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np

def load_data_withoutHeader(file_path):
    df = pd.read_csv(file_path, header=None)
    data = df.to_numpy()
    X = data[:, :-1].T  
    T = data[:, -1].reshape(1, -1)  
    return X, T

def load_data(nome_ficheiro):
    df = pd.read_csv(nome_ficheiro)
    return df

# Função de ativação sigmoidal
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Inicialização de pesos e bias
def initialize_weights(nx, nh1, ny,n):
    W1 = np.random.random((nh1, nx))
    B1 = np.random.random((nh1, 1))
    W2 = np.random.random((ny, nh1))
    B2 = np.random.random((ny, 1))
    MI = np.ones((n,1))
    return W1, B1, W2, B2, MI

# Passagem forward
def forward_pass(X, W1, B1, W2, B2):
    a0 = X
    a1 = np.dot(W1, a0) + B1
    y1 = sigmoid(a1)
    a2 = np.dot(W2, a1) + B2
    y2 = a2  # camada de saída linear
    return a0, a1, y1, a2, y2

def compute_error(T, y2):
    e2 = T - y2
    erro = np.dot(e2, e2.T)
    return e2, erro

def backpropagation(W1,B1, B2,W2,a1, alfa, e2, y1, a0, MI):
    
    dy2 = 1 
    S2 = e2 * dy2
    dW2 = alfa * np.dot(S2, a1.T)
    dB2 = alfa * np.dot(S2, MI)
        
    W2 = W2 + dW2
    B2 = B2 + dB2
        
    dy1 = y1 * (1 - y1)
    S1 = np.dot(W2.T, S2) * dy1
    dW1 = alfa * np.dot(S1, a0.T)
    dB1 = alfa * np.dot(S1, MI)
    
    W1 = W1 + dW1
    B1 = B1 + dB1
    
    return W1, B1, W2, B2
    
def train_network(X, T, EPOCHS, alfa, W1, B1, W2, B2, MI):
    
    for numEP in range(EPOCHS):
        
        a0, a1, y1, a2, y2 = forward_pass(X, W1, B1, W2, B2)
        print(y2)
        
        e2, erro = compute_error(T, y2)
        
        W1, B1, W2, B2 = backpropagation(W1, B1, B2, W2, a1, alfa, e2, y1, a0, MI)
        
        if numEP % 100 == 0:
            print(f'Epoch {numEP}, Error: {erro[0, 0]}')
    
    return W1, B1, W2, B2

if __name__ == "__main__":
    X,T = load_data_withoutHeader("P4_function.csv")
    # Parâmetros iniciais
    alfa = 0.001
    EPOCHS = 1000
    nx = 2  # Número de entradas
    nh1 = 3  # Número de neurônios na camada oculta
    ny = 1  # Número de saídas

    # Inicialização dos pesos e bias
    W1, B1, W2, B2, MI = initialize_weights(nx, nh1, ny, X.shape[1])
    
    # Treinando a rede
    W1, B1, W2, B2 = train_network(X, T, EPOCHS, alfa, W1, B1, W2, B2, MI)

    
    
        