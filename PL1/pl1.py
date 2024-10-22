import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(nome_ficheiro):
    df = pd.read_csv(nome_ficheiro)
    D = df.values
    X = D[:, 0:6]  #input
    T = D[:, 6]  #Target
    N =  X.shape[0] #numero de pacientes

    return df, X, T, N

def pre_processing(X, N):
    valid_values = X[X[:, 1] != -1, 1] # Seleciona os valores que não são -1
    total_average = np.mean(valid_values).astype(int)  # Calcula a média dos valores válidos, e converte para inteiro, pois a idade é um inteiro 
    X[X[:, 1] == -1, 1] = total_average   # Substitui os valores -1 pela média calculada


def visualize_data(df,T, X):
    df.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False)
    plt.show()

    X_no_event = X[T== 0]
    X_event = X[T== 1]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))   # Cria uma figura com 1 linha e 2 colunas de subplots

    axs[0].scatter(X_no_event[:, 1], X_no_event[:, 2], c='b', marker='+', label='No Event')
    
    axs[0].scatter(X_event[:, 1], X_event[:, 2], c='r', marker='o', label='Event')
    
    axs[0].set_xlabel('Age')
    axs[0].set_ylabel('Heart Rate')
    axs[0].set_title('Age vs Heart Rate (No Event vs Event)')
    axs[0].legend()

  
    axs[1].scatter(X_no_event[:, 1], X_no_event[:, 3], c='b', marker='+', label='No Event')
    
    axs[1].scatter(X_event[:, 1], X_event[:, 3], c='r', marker='o', label='Event')
    
    axs[1].set_xlabel('Age')
    axs[1].set_ylabel('SBP')
    axs[1].set_title('Age vs SBP (No Event vs Event)')
    axs[1].legend()

    plt.tight_layout() # Ajusta o layout dos subplots
    plt.show()


def classification_model(X, T, N):
    #Mean of patiens with attributes T = 0
    X_no_event = X[T == 0]
    mean_no_event = np.mean(X_no_event)
    X_event = X[T == 1]
    mean_event = np.mean(X_event)

    d0_list = []
    d1_list = []
    labels = []

    # Classifying patients based on distances d0 and d1
    for i in range(N):
        d0 = np.linalg.norm(X[i] - mean_no_event) #Calcula a distância euclidiana entre o paciente i e a média dos pacientes sem evento
        d1 = np.linalg.norm(X[i] - mean_event) #Calcula a distância euclidiana entre o paciente i e a média dos pacientes com evento
        d0_list.append(d0)
        d1_list.append(d1)
        
        if d0 < d1:
            labels.append('No Event')
           # print(f'Patient {i+1} is classified as No Event')
        else:
            labels.append('Event')
            #print(f'Patient {i+1} is classified as Event')

    # Create scatter plot of distances d0 vs d1
    plt.figure(figsize=(8, 6))
    for i in range(N):
        color = 'blue' if labels[i] == 'No Event' else 'orange'
        plt.scatter(d0_list[i], d1_list[i], color=color, label=labels[i] if i == 0 else "")

    plt.plot([0, max(d0_list)], [0, max(d1_list)], 'k--', label='d0 = d1 boundary')
    plt.xlabel('Distance to No Event (d0)')
    plt.ylabel('Distance to Event (d1)')
    plt.title('Classification of Patients: Event vs No Event')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

def evaluate_model(T, N):
    TP = np.sum(T == 1)
    TN = np.sum(T == 0)
    FP = N - TN
    FN = N - TP

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')




if __name__ == "__main__":
    df, X, T, N = load_data('C:\\Users\\User\\Desktop\\MEB\\1Semestre\\AC\\PL\\P1_cardiacRisk.csv')
    #print(df.describe())
    pre_processing(X,N)
    visualize_data(df,T, X)
    classification_model(X, T,N)
    evaluate_model(T, N)
