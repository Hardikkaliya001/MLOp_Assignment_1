from sklearn.kernel_ridge import KernelRidge
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

def load_data():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    return pd.DataFrame(X), y

def preprocess(df, y=None, scale=False):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def save_model(model, filename):
    joblib.dump(model, filename)

def main():
    df, y = load_data()  
    X_train, X_test, y_train, y_test, scaler = preprocess(df, y, scale=True)  
    model = KernelRidge(alpha=1.0, kernel='rbf')
    model = train_model(model, X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)
    print(f"KernelRidge MSE on test set: {mse:.4f}")
    save_model(model, "kernel_ridge_model.joblib")

if __name__ == "__main__":
    main()