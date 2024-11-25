# Individuals with suspected COVID are admitted to the hospital emergency room
# At the time of admission, several variables/parameters are acquired (low cost and simple to acquire)
# Based on these variables, the health professional must decide whether the individual remains hospitalized for additional examinations or should return home

# COVID_numerics.csv contains the following variables:
# Screening (1 | Gender; 2 | Age; 3 | Mariatal status; 4 | Vaccinated; 5 | Breathing difficulty)
# Measurements (6 | Heart rate; 7 | Blood pressure) and 8 | Temperature
# knowledge ( If breathing difficulty >= moderate and temperature >= 37.8, then Stay at hospital)

#COVID_IMG.csv contains the following variables:
# ECG - Phase space plot
# Matrix (21,21)
# Binary values {0,1}
# These plot/image can reveal patterns in the ECG data, such as periodicity or anomalies

#  Design a machine learning model to address this issue:
# ▪ Decide whether the individual should remain hospitalized for additional examinations or be discharged to return home.

# The model should be able to:
# ▪ Predict the outcome of the decision based on the variables in the COVID_numerics.csv file and the image in the COVID_IMG.csv file.


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
def load_data():
    # Load the COVID_numerics.csv file
    colunas = ["GENDER","AGE","MARITAL STATUS","VACINATION","RESPIRATION CLASS","HEART RATE","SYSTOLIC BLOOD PRESSURE","TEMPERATURE","TARGET"]
    df_numerics = pd.read_csv('COVID_numerics.csv', usecols=colunas)
    # Load the COVID_IMG.csv file without header
    df_img = pd.read_csv('COVID_IMG.csv', header=None)
    return df_numerics, df_img


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
    
    
    
    return df_numerics, df_img
    


def main():
    # Load the data
    df_numerics, df_img = load_data()
    # First lines of the data numerics
    print(df_numerics.head())
    # Format of the data image
    print(df_img.shape)
    # Preprocess the data
    df_numerics, df_img = preprocess_data(df_numerics, df_img)
    

if __name__ == '__main__':
    main()
