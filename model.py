import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("StudentsPerformance.csv")

def load_data(csv_path: str):

    print("Dataset loaded Successfully")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    return df

def visualize_data(df):
    
    #Histrogram
    plt.hist(df["math score"],bins=20)
    plt.xlabel("Math Score")
    plt.ylabel("Number of Students")
    plt.title("Distribution of Math Score")
    plt.show()
    print(df["math score"].max())

    #scatter plot
    plt.scatter(df['reading score'],df['math score'] ,s=10)
    plt.xlabel('Reading Score')
    plt.ylabel('Math Score')
    plt.title('Reading Score vs Math Score')
    plt.show()

    # Scatter : writing vs math
    plt.scatter(df['writing score'],df['math score'],s=10)
    plt.xlabel('Writing Score')
    plt.ylabel('Math Score')
    plt.title("Writing Score vs Math Score")
    plt.show()


visualize_data(df)
load_data("StudentsPerformance.csv")
