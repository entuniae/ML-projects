import numpy as np
import pandas as pd

PATH = "mushrooms.csv"
df = pd.read_csv(PATH)

#0------------------------------------------
# Inspecting the data
print(df.shape)
print(df.head())
print(df.dtypes)
print(df["class"].value_counts())
print(df.isna().sum().sort_values(ascending=False).head(10))

#-------------------------------------------
# CLEANING & PREPROCESSING
#-------------------------------------------

#1---------
# Clean Data from missing 
# (Not really needed here since there are no missing values, but good practice to check)
# Drop columns with too many missing values (if any)
df = df.replace("?", np.nan)   # If data uses "?" to indicate missing values, replace them with NaN
df = df.dropna()               # Delete rows with missing values

#2----------
# Split data 

np.random.seed(42) 
# Sets a random seed for reproducibility, so that the same random operations yield the same results each time you run the code

df = df.sample(frac=1).reset_index(drop=True)  
# Sample(frac=1) → randomly shuffles all rows in the DataFrame
# Reset_index(drop=True) → creates a new index after shuffling and drops the old index to avoid confusion

split_index = int(0.8 * len(df))  
# Calculates the index that corresponds to 80% of the data, which will be used to split the DataFrame into training and test sets

train_df = df[:split_index]     
# First 80% of the rows → traindata

test_df = df[split_index:]      
# Last 20% of the rows → testdata

print("Train distribution:")    
print(train_df["class"].value_counts(normalize=True))  
# Prints the distribution of classes in the training set, showing the proportion of each class (e.g., edible vs poisonous) in the training data. 
# This helps to check if the classes are balanced or if there is any imbalance that might affect model training.

print("\nTest distribution:")   
print(test_df["class"].value_counts(normalize=True))  
# Prints the distribution of classes in the test set, similar to the training set, 
# to ensure that the test data also has a representative distribution of classes for evaluating the model's performance.


#3-----------
# X-train, y-train, X-test, y-test
# Target (what we want to predict)
y_train = train_df["class"] # only the "class" column, which indicates whether the mushroom is edible or poisonous
y_test = test_df["class"]

# Features (all columns except class)
X_train = train_df.drop(columns=["class"])
X_test = test_df.drop(columns=["class"])

# Check shapes
print(X_train.shape)
print(y_train.shape)


#4-------------
# One-hot encoding: categorical data needs to be converted into a numerical format that machine learning models can understand.
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Make sure both train and test have the same columns (features) after encoding
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

print("X_train shape:", X_train_encoded.shape)
print("X_test shape:", X_test_encoded.shape)


#-------------------------------------------
# BUILDING A MODEL
#-------------------------------------------