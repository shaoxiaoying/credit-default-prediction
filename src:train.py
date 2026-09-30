import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, get_preprocessor

def train_pipeline(data_path: str, model_type: str = 'logistic_regression'):
    """Trains a credit risk classification model pipeline and exports model weights."""
    df = load_data(data_path)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = get_preprocessor(X_train)
    
    if model_type == 'logistic_regression':
        classifier = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    # Save trained pipeline
    model_filename = f"model_{model_type}.joblib"
    joblib.dump(model_pipeline, model_filename)
    print(f"Model successfully trained and saved to {model_filename}")
    
    return model_pipeline, X_test, y_test

if __name__ == '__main__':
    train_pipeline('../data/german_credit_data.csv', model_type='logistic_regression')