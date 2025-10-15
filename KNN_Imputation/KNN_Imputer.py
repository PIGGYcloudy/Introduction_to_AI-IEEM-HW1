import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import os

def imputer(n_neighbors, train_df, test_df, writefile=False, outputfolder='./'):
    train_copy = train_df.copy()
    test_copy = test_df.copy()

    train_ID = train_copy['Employee ID']
    test_ID = test_copy['Employee ID']

    train_target = train_copy['Burn Rate']
    train_features = train_copy.drop(['Burn Rate', 'Employee ID'], axis=1)

    test_features = test_copy.drop(['Employee ID'], axis=1)
    train_rows = len(train_features)

    combined_df = pd.concat([train_features, test_features], ignore_index=True)
    df_numeric = combined_df.copy()
    df_numeric['Date of Joining'] = pd.to_datetime(df_numeric['Date of Joining'])
    reference_date = df_numeric['Date of Joining'].min()
    df_numeric['Date of Joining'] = (df_numeric['Date of Joining'] - reference_date).dt.days
    
    map = {}
    for col in ['Gender', 'Company Type', 'WFH Setup Available']:
        unique_vals = df_numeric[col].dropna().unique()
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            map[col] = mapping
            df_numeric[col] = df_numeric[col].map(mapping)

    scaler = MinMaxScaler()
    df_columns = df_numeric.columns
    df_index = df_numeric.index

    df_scaled = scaler.fit_transform(df_numeric)
    
    _imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed_scaled = _imputer.fit_transform(df_scaled)
    df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=df_columns, index=df_index)
    df_restored = df_imputed.copy()
    
    df_restored['Date of Joining'] = reference_date + pd.to_timedelta(df_restored['Date of Joining'].round(), unit='D')

    df_restored['Date of Joining'] = df_restored['Date of Joining'].dt.strftime('%Y-%m-%d')

    for col in ['Gender', 'Company Type', 'WFH Setup Available']:
        inverse_mapping = {v: k for k, v in map[col].items()}
        df_restored[col] = df_restored[col].round().astype(int).map(inverse_mapping)
    df_restored[['Designation', 'Resource Allocation']] = df_restored[['Designation', 'Resource Allocation']].round().astype(int)
    
    
    final_train_features = df_restored.iloc[:train_rows]
    final_test_features = df_restored.iloc[train_rows:]

    # 關鍵！重設兩邊的索引，讓它們都能從 0 開始對齊
    # drop=True 表示不要把舊的 index 變成一個新的 column
    final_train_features.reset_index(drop=True, inplace=True)
    final_test_features.reset_index(drop=True, inplace=True)
    train_ID.reset_index(drop=True, inplace=True)
    test_ID.reset_index(drop=True, inplace=True)

    # 現在可以安全地合併
    train_imputed_df = pd.concat([train_ID, final_train_features], axis=1)
    train_imputed_df['Burn Rate'] = train_target

    test_imputed_df = pd.concat([test_ID, final_test_features], axis=1)
    
    
    if writefile:
        os.makedirs(outputfolder, exist_ok=True)
        train_imputed_df.to_csv(os.path.join(outputfolder, 'train_imputed.csv'), index=False)
        test_imputed_df.to_csv(os.path.join(outputfolder, 'test_imputed.csv'), index=False)
    return train_imputed_df, test_imputed_df

if __name__ == '__main__':
    n_neighbors = 5

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    train_df, test_df = imputer(n_neighbors, train_df, test_df, writefile=True, outputfolder='./KNN_Imputation')
