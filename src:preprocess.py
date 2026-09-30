import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    """Loads dataset and encodes target variable (bad=1 [default], good=0)."""
    df = pd.read_csv(filepath)
    df['target'] = df['target'].map({'bad': 1, 'good': 0})
    return df

def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Constructs preprocessing pipeline for numeric and categorical columns."""
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )
    return preprocessor