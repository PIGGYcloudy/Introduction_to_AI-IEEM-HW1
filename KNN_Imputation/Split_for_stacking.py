import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./KNN_Imputation/train_imputed.csv")

target_col = "Burn Rate"

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42 # 確保每次分割結果都相同
)

train_df_new = pd.concat([X_train, y_train], axis=1)

stacking_valid_df = pd.concat([X_holdout, y_holdout], axis=1)

train_output_filename = './KNN_Imputation/train_imputed_stacking.csv'
valid_output_filename = './KNN_Imputation/validation_imputed_stacking.csv'

train_df_new.to_csv(train_output_filename, index=False)
stacking_valid_df.to_csv(valid_output_filename, index=False)