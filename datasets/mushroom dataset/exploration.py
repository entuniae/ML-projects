import numpy as np
import pandas as pd

PATH = "mushrooms.csv"
df = pd.read_csv(PATH)

# Inspect data
print(df.shape)
print(df.head())
print(df.dtypes)
print(df["class"].value_counts())
print(df.isna().sum().sort_values(ascending=False).head(10))

#-------------------------------------------
# CLEANING & PREPROCESSING
#-------------------------------------------

#1---------
# Clean Data from missing (not really needed in this dataset)
df = df.replace("?", np.nan)   # Replace ? with NaN
df = df.dropna()               # Delete rows with missing values

#2----------
# Split data 

# Set random seed for reproducibility
np.random.seed(42) 

# randly shuffle the DataFrame, creates a new index after shuffling and drops the old index to avoid confusion
df = df.sample(frac=1).reset_index(drop=True)  

# Split 80/20
split_index = int(0.8 * len(df))  
train_df = df[:split_index]  # First 80% of the rows → traindata
test_df = df[split_index:]   # Last 20% of the rows → testdata

# Check distribution
print("Train distribution:")    
print(train_df["class"].value_counts(normalize=True))  

print("\nTest distribution:")   
print(test_df["class"].value_counts(normalize=True))  


#3-----------
# X-train, y-train, X-test, y-test
y_train = train_df["class"] # only the "class" column, which indicates whether the mushroom is edible or poisonous
y_test = test_df["class"]

# Features (all columns except class)
X_train = train_df.drop(columns=["class"])
X_test = test_df.drop(columns=["class"])

# Check shapes
print(X_train.shape)
print(y_train.shape)


#4-------------
# One-hot encoding: categorial to numerical encoding
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Make sure both train and test have the same columns (features) after encoding
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Check shapes after encoding
print("X_train shape:", X_train_encoded.shape)
print("X_test shape:", X_test_encoded.shape)


#-------------------------------------------
# LOGISTIC REGRESSION
#-------------------------------------------

# Gör om target till binär
y_train_bin = (y_train == "p").astype(int).to_numpy()
y_test_bin = (y_test == "p").astype(int).to_numpy()

Xtr = X_train_encoded.to_numpy(dtype=float)
Xte = X_test_encoded.to_numpy(dtype=float)

# Lägg till intercept (bias-term)
Xtr = np.c_[np.ones(Xtr.shape[0]), Xtr]
Xte = np.c_[np.ones(Xte.shape[0]), Xte]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, epochs=300):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    for _ in range(epochs):
        z = X @ w
        p = sigmoid(z)
        gradient = (X.T @ (p - y)) / n_samples
        w -= lr * gradient

    return w

w = train_logistic_regression(Xtr, y_train_bin, lr=0.5, epochs=400)

def predict_proba(X, w):
    return sigmoid(X @ w)

def predict(X, w, threshold=0.5):
    return (predict_proba(X, w) >= threshold).astype(int)

y_pred = predict(Xte, w)

accuracy = np.mean(y_pred == y_test_bin)
print("Accuracy:", accuracy)


#-------------------------------------------
# EVALUATION
#-------------------------------------------