from sklearn.tree import DecisionTreeRegressor
import numpy as np
import joblib  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd  


def load_data():
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.rand(100)
    }
    return pd.DataFrame(data)  

def preprocess(df, scale=False):
    # Example preprocessing
    X = df.drop('target', axis=1)  
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, None  

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

def save_model(model, filename):
    joblib.dump(model, filename)

def main():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df, scale=False)  
    model = DecisionTreeRegressor(random_state=42)
    model = train_model(model, X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)
    print(f"DecisionTreeRegressor MSE on test set: {mse:.4f}")
    save_model(model, "decision_tree_model.joblib")

if __name__ == "__main__":
    main()