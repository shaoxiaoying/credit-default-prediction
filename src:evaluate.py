import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from preprocess import load_data
from sklearn.model_selection import train_test_split

def evaluate_model(model_path: str, data_path: str):
    """Evaluates serialized model pipeline on holdout test set."""
    df = load_data(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = joblib.load(model_path)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("--- Model Evaluation Metrics ---")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good (0)', 'Default (1)']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    evaluate_model('model_logistic_regression.joblib', '../data/german_credit_data.csv')