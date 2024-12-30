from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
from PIL import Image

# Load the data
def load_data():
    colunas = ["GENDER", "AGE", "MARITAL STATUS", "VACINATION", "RESPIRATION CLASS", "HEART RATE", "SYSTOLIC BLOOD PRESSURE", "TEMPERATURE", "TARGET"]
    df_numerics = pd.read_csv('COVID_numerics.csv', usecols=colunas)
    df_img = pd.read_csv('COVID_IMG.csv', header=None)
    return df_numerics, df_img

# Add rule for feature engineering
def add_rule(X):
    X['RULE'] = ((X["RESPIRATION CLASS"] >= 2) & (X["TEMPERATURE"] > 37.8)).astype(int)
    return X

# Preprocess the tabular data
def preprocess_cnn(df_numerics):
    df_numerics = add_rule(df_numerics)
    df_numerics.drop_duplicates(inplace=True)
    df_numerics.fillna(df_numerics.mean(), inplace=True)

    continuous_columns = ["AGE", "HEART RATE", "SYSTOLIC BLOOD PRESSURE", "TEMPERATURE"]
    scaler = MinMaxScaler()
    df_numerics[continuous_columns] = scaler.fit_transform(df_numerics[continuous_columns])

    print("Correlation between variables")
    corr = df_numerics.corr()
    sns.heatmap(corr, annot=True)
    plt.show()

    df_numerics = pd.get_dummies(df_numerics, columns=["GENDER", "MARITAL STATUS", "VACINATION", "RESPIRATION CLASS"], drop_first=True)

    target = df_numerics["TARGET"]
    X = df_numerics.drop(columns=["TARGET"])
    
    return X, target

# Resize image data to uniform size
def resize_images(df_img, size=(28, 28)):
    resized_images = []
    for img in df_img.values:
        img_reshaped = img.reshape(int(np.sqrt(len(img))), -1)
        img_resized = np.array(Image.fromarray(img_reshaped).resize(size))
        resized_images.append(img_resized.flatten())
    return np.array(resized_images)

# Process image data
def process_img_data(df_img):
    X_img = resize_images(df_img, size=(28, 28))
    X_img = X_img / 255.0
    X_img = X_img.reshape(-1, 1, 28, 28)  # Shape [batch_size, channels, height, width]
    return X_img

# Combine tabular and image data
def combine_features(X_tabular, X_img):
    X_img_flattened = X_img.reshape(X_img.shape[0], -1)
    X_combined = np.hstack((X_tabular, X_img_flattened))
    return X_combined

# Combined Model
class CombinedModel(nn.Module):
    def __init__(self, tabular_input_dim):
        super(CombinedModel, self).__init__()
        # Image path
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_img = nn.Linear(16 * 14 * 14, 128)
        
        # Tabular path
        self.fc_tabular = nn.Linear(tabular_input_dim, 128)
        
        # Combined path
        self.fc_combined = nn.Linear(128 + 128, 2)

    def forward(self, x_tabular, x_img):
        # Image path
        x_img = self.pool(F.relu(self.conv1(x_img)))
        x_img = x_img.view(-1, 16 * 14 * 14)
        x_img = F.relu(self.fc_img(x_img))
        
        # Tabular path
        x_tabular = F.relu(self.fc_tabular(x_tabular))
        
        # Combine
        x = torch.cat((x_tabular, x_img), dim=1)
        x = self.fc_combined(x)
        return x

# Train the model
def train_cnn(X_tabular, X_img, target):
    # Divida os dados em treinamento e teste
    Xtrain_tab, Xtest_tab, Xtrain_img, Xtest_img, ytrain, ytest = train_test_split(
        X_tabular, X_img, target, test_size=0.2, random_state=42
    )

    # Converter os dados para tipos numéricos compatíveis
    Xtrain_tab = Xtrain_tab.astype(np.float32)  # Converta para float32
    Xtest_tab = Xtest_tab.astype(np.float32)
    ytrain = ytrain.astype(np.int64)  # Converta o alvo para int64

    # Definir o modelo
    model = CombinedModel(tabular_input_dim=Xtrain_tab.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Converter dados para tensores
    Xtrain_tab = torch.tensor(Xtrain_tab, dtype=torch.float32)
    Xtrain_img = torch.tensor(Xtrain_img, dtype=torch.float32)
    ytrain = torch.tensor(ytrain, dtype=torch.long)

    # Loop de treinamento
    for epoch in range(10):  # Número de épocas reduzido para testes
        optimizer.zero_grad()
        outputs = model(Xtrain_tab, Xtrain_img)
        loss = criterion(outputs, ytrain)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    
    return model, Xtest_tab, Xtest_img, ytest

# Evaluate the model
def evaluate_model(model, Xtest_tab, Xtest_img, ytest):
    # Converter os dados para tensores
    Xtest_tab = torch.tensor(Xtest_tab, dtype=torch.float32)
    Xtest_img = torch.tensor(Xtest_img, dtype=torch.float32)
    
    # Obter as previsões do modelo
    model.eval()
    with torch.no_grad():
        outputs = model(Xtest_tab, Xtest_img)
        ypred = torch.argmax(outputs, dim=1).detach().numpy()

    # Calcular a matriz de confusão
    cm = confusion_matrix(ytest, ypred)
    TN, FP, FN, TP = cm.ravel()

    # Calcular métricas
    accuracy = accuracy_score(ytest, ypred)
    precision = precision_score(ytest, ypred)
    recall = recall_score(ytest, ypred)  # Também conhecido como sensibilidade
    specificity = TN / (TN + FP)
    f1 = f1_score(ytest, ypred)
    roc_auc = roc_auc_score(ytest, outputs[:, 1])  # Para ROC-AUC, usamos os scores antes de aplicar argmax

    # Exibir os resultados
    print("Matriz de Confusão:")
    print(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    print("\nRelatório de Classificação:")
    print(classification_report(ytest, ypred, digits=4))

    print("Métricas detalhadas:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensibilidade): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Plotar a curva ROC
    fpr, tpr, _ = roc_curve(ytest, outputs[:, 1])  # Usamos as probabilidades para a curva ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Linha de referência
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Main execution
def main():
    df_numerics, df_img = load_data()
    X_tabular, target = preprocess_cnn(df_numerics)
    X_img = process_img_data(df_img)
    model, Xtest_tab, Xtest_img, ytest = train_cnn(X_tabular.values, X_img, target)
    evaluate_model(model, Xtest_tab, Xtest_img, ytest)

if __name__ == "__main__":
    main()
