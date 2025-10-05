import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

def load_data():
    """
    Load the Boston dataset from the CMU statlib URL as in assignment instructions.
    Returns pandas DataFrame with MEDV target column.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def preprocess(df, target_col='MEDV', test_size=0.2, random_state=42, scale=True):
    """
    Generic preprocessing:
     - Splits into X_train, X_test, y_train, y_test
     - Optionally scales features (StandardScaler)
    Returns: X_train, X_test, y_train, y_test, scaler (or None)
    """
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def train_model(model, X_train, y_train):
    """Train a scikit-learn model and return the fitted model"""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model: compute MSE on the test set and return it.
    """
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

def save_model(model, path):
    """Save model to disk using joblib"""
    joblib.dump(model, path)

def load_model(path):
    """Load model from disk"""
    return joblib.load(path)
    df, y = load_data()  
    X_train, X_test, y_train, y_test, scaler = preprocess(df, y, scale=True)  
    model = KernelRidge(alpha=1.0, kernel='rbf')
    model = train_model(model, X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)
    print(f"KernelRidge MSE on test set: {mse:.4f}")
    save_model(model, "kernel_ridge_model.joblib")

if __name__ == "__main__":
    main()