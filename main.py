import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(path):
    return pd.read_csv(path)

def eda(df, output_dir='images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Data Head:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Pairplot
    sns.pairplot(df, hue='species')
    plt.savefig(f"{output_dir}/pairplot.png")
    plt.clf()
    
    # Correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.clf()

def preprocess(df):
    X = df.drop('species', axis=1)
    y = df['species']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("images/confusion_matrix.png")
    plt.clf()

def main():
    data_path = 'data/sample_data.csv'
    df = load_data(data_path)
    eda(df)
    X, y, le = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, le)

if __name__ == "__main__":
    main()
