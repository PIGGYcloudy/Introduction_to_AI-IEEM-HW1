import pandas as pd
import joblib
from pathlib import Path


def preprocess_data(df, current_dir):
    df = df.dropna()

    df['Date of Joining'] = pd.to_datetime(df['Date of Joining'])
    reference_date = pd.to_datetime(pd.read_csv(current_dir.parent / 'train.csv')['Date of Joining']).min()
    df['Days_with_company'] = (df['Date of Joining'] - reference_date).dt.days
    
    categorical_features = ['Gender', 'Company Type', 'WFH Setup Available']

    for col in categorical_features:
        df[col] = df[col].astype('category')
    df = df[[
        'Designation', 'Resource Allocation',
        'Mental Fatigue Score', 'Days_with_company',
        'Gender', 'Company Type', 'WFH Setup Available'
    ]]
    return df

def inference(df):
    current_dir = Path(__file__).parent
    df = preprocess_data(df, current_dir)
    
    model_path = current_dir / 'stacking_lgbm_model.joblib'
    
    model = joblib.load(model_path)
    return model.predict(df)
