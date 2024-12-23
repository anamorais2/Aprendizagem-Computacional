import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
# Preprocessing the data
def preprocess_data(df_numerics, df_img):
    # Verificar se ha valores null e substituir pela media
    df_numerics.isnull().sum()
    df_numerics.fillna(df_numerics.mean(), inplace=True) # substituir os valores null pela media
    
    # Outliers
    # Verificar se ha outliers
    sns.boxplot(data=df_numerics)
    plt.show()
    # Remover os outliers
    continuous_columns = ['AGE', 'HEART RATE', 'SYSTOLIC BLOOD PRESSURE', 'TEMPERATURE']
    
    for column in continuous_columns:
        Q1 = df_numerics[column].quantile(0.25)
        Q3 = df_numerics[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_numerics = df_numerics[(df_numerics[column] >= lower_bound) & (df_numerics[column] <= upper_bound)]
   
   # Verificar se ha outliers
    sns.boxplot(data=df_numerics)
    plt.show()
    
    # Normalização
    scaler = MinMaxScaler()
    df_numerics[continuous_columns] = scaler.fit_transform(df_numerics[continuous_columns])
    
    #Correlação
    # Verificar a correlação entre as variáveis
    print("Correlation between variables") 
    corr = df_numerics.corr()
    sns.heatmap(corr, annot=True)
    plt.show()
    
    # Tentar uma correlao não linear
    # ás vezes uma correlação de o.2 já é suficiente para uma boa predição, temos de olhar de uma forma positiva. 
    # A correlação linear não é a única forma de medir a relação entre variáveis.
    
    print("Valor -0.4 entre vacination and target, ou seja quando maior a vacinação menor a probabilidade de ficar no hospital")
    #Mas é possivel pegar na informação que a correlação dá e tirar algum conclusão prática?
    #Sim, podemos concluir que a vacinação tem um impacto na probabilidade de ficar no hospital
    
    return df_numerics, df_img
