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

# Preprocessing and regression pipeline

#                    ┌─────────────────────────────┐
#                    │       INPUT FEATURES         │
#                    │  (X: DataFrame with columns) │
#                    └──────────────┬──────────────┘
#                                   │
#         ┌─────────────────────────┼─────────────────────────┐
#         │                         │                         │
#         ▼                         ▼                         ▼

# ┌──────────────────────┐   ┌──────────────────────┐   (Optional)
# │ Categorical Columns   │   │   Numeric Columns    │
# │ e.g., gender, race    │   │ e.g., math score     │
# └───────────┬──────────┘   └───────────┬──────────┘
#             │                          │
#             │                          │
#             ▼                          ▼

# ┌──────────────────────┐   ┌──────────────────────┐
# │ OneHotEncoder         │   │   Passthrough        │
# │ drop="first"          │   │ (kept as-is)         │
# │ handle_unknown="ignore"│  │                       │
# └───────────┬──────────┘   └───────────┬──────────┘
#             │                          │
#             └──────────┬───────────────┘
#                        ▼

#             ┌───────────────────────────────┐
#             │  Combined Preprocessed Output  │
#             │  (All data now numeric)        │
#             └───────────────┬───────────────┘
#                             ▼

#             ┌───────────────────────────────┐
#             │      Linear Regression        │
#             │  (Trains on processed data)   │
#             └───────────────┬───────────────┘
#                             ▼

#             ┌──────────────────────────────┐
#             │         FINAL MODEL          │
#             │    model.fit() / predict()   │
#             └──────────────────────────────┘

def build_pipeline(categorical_cols,numeric_cols):
    preprocessor=ColumnTransformer(
        transformers=[
            ("cat",OneHotEncoder(drop='first',handle_unknown='ignore'),categorical_cols),
            ("num",'passthrough',numeric_cols),
        ]
    )
    model=Pipeline(steps=[
        ("preprocess",preprocessor),
        ("regreesor",LinearRegression())
    ])
    return model

# training the model and evaluating it

def train_and_evaluate(df):
    x=df[[
        'gender',
        'race',
        'parental level of education',
        'lunch',
        'test preparation course',
        'reading score',
        'writing score'
    ]]
    y=df['math score']
    categorical_cols=[
        'gender',
        'race',
        'parental level of education',
        'lunch',
        'test preparation course',
    ]
    numeric_cols=['reading score','writing score']
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    


visualize_data(df)
load_data("StudentsPerformance.csv")
