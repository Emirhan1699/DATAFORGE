import pandas as pd
import numpy as np
import xgboost as xgb
import optuna

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)

print("Veriler yukleniyor...")

# veri

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "Production"

# Hedefi bos olanlari siliyoruz
train = train.dropna(subset=[TARGET])

# Crop_Value_Index bosluklarini ortanca ile dolduruyoruz
train_median = train['Crop_Value_Index'].median()
train['Crop_Value_Index'] = train['Crop_Value_Index'].fillna(train_median)
test['Crop_Value_Index'] = test['Crop_Value_Index'].fillna(train_median)

# Hedefin Logaritmasini aliyoruz
y = np.log1p(train[TARGET])

# X icinden hedefi cikariyoruz
X = train.drop(columns=[TARGET])
X_test = test.copy()

test_ids = X_test["ID"]

if "ID" in X.columns:
    X = X.drop(columns=["ID"])
if "ID" in X_test.columns:
    X_test = X_test.drop(columns=["ID"])


# feature engineering 

print("Ozellik muhendisligi yapiliyor...")

def process_year(df):
    try:
        df["Year_Start"] = df["Year"].str.split("-").str[0].astype(int)
        df["Year_End"] = df["Year"].str.split("-").str[1].astype(int)
        df["Year_Diff"] = df["Year_End"] - df["Year_Start"]
    except:
        pass
    return df

X = process_year(X)
X_test = process_year(X_test)

X["Log_Area"] = np.log1p(X["Area"])
X_test["Log_Area"] = np.log1p(X_test["Area"])

for col in ["State", "District", "Crop"]:
    freq = X[col].value_counts()
    X[col+"_freq"] = X[col].map(freq)
    X_test[col+"_freq"] = X_test[col].map(freq)

for col in X.select_dtypes("object"):
    X[col] = X[col].fillna("Unknown")
    X_test[col] = X_test[col].fillna("Unknown")

# XGBoost icin Ortak Sozluk ile Kategorik Veri Hazirligi
cat_cols = X.select_dtypes("object").columns
for col in cat_cols:
    tum_kategoriler = pd.concat([X[col], X_test[col]]).unique()
    ortak_tip = pd.CategoricalDtype(categories=tum_kategoriler, ordered=False)
    
    X[col] = X[col].astype(ortak_tip)
    X_test[col] = X_test[col].astype(ortak_tip)

# Sutun siralamasi garantisi
X_test = X_test[X.columns]

# optuna

print("Optuna hiperparametre aramasi basliyor...")

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42,
        "n_jobs": -1
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)

    preds = np.where(np.expm1(model.predict(X_val)) < 0, 0, np.expm1(model.predict(X_val)))
    true = np.expm1(y_val)
    mse = mean_squared_error(true, preds)

    return mse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

best_params = study.best_params
best_params.update({"tree_method": "hist", "enable_categorical": True, "random_state": 42, "n_jobs": -1})
print("\nEn Iyi Parametreler:", best_params)

# K-FOLD 

print("\nFinal egitimi (5-Fold CV) ve Tutarlilik Kontrolu basliyor...")

kf_final = KFold(n_splits=5, shuffle=True, random_state=42)
test_predictions = np.zeros(len(X_test))

train_mse_scores = []
val_mse_scores = []
train_rmse_scores = []
val_rmse_scores = []

for fold, (tr_idx, val_idx) in enumerate(kf_final.split(X)):
    print(f"\n--- Fold {fold+1} ---")
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])

    # 1. Egitim (Train) Skorlarini Hesapla
    train_preds = np.where(np.expm1(model.predict(X.iloc[tr_idx])) < 0, 0, np.expm1(model.predict(X.iloc[tr_idx])))
    true_train = np.expm1(y.iloc[tr_idx])
    
    train_mse = mean_squared_error(true_train, train_preds)
    train_rmse = np.sqrt(train_mse)
    
    train_mse_scores.append(train_mse)
    train_rmse_scores.append(train_rmse)

    # 2. Test (Validation) Skorlarini Hesapla
    val_preds = np.where(np.expm1(model.predict(X.iloc[val_idx])) < 0, 0, np.expm1(model.predict(X.iloc[val_idx])))
    true_val = np.expm1(y.iloc[val_idx])
    
    val_mse = mean_squared_error(true_val, val_preds)
    val_rmse = np.sqrt(val_mse)
    
    val_mse_scores.append(val_mse)
    val_rmse_scores.append(val_rmse)

    print(f"Train MSE : {train_mse:,.2f} | Train RMSE : {train_rmse:,.2f}")
    print(f"Test MSE  : {val_mse:,.2f} | Test RMSE  : {val_rmse:,.2f}")

    # Teslimat icin tahminleri biriktir
    fold_preds = np.where(np.expm1(model.predict(X_test)) < 0, 0, np.expm1(model.predict(X_test)))
    test_predictions += fold_preds / kf_final.n_splits

print(f"ORTALAMA TRAIN MSE  : {np.mean(train_mse_scores):,.2f}")
print(f"ORTALAMA TEST MSE   : {np.mean(val_mse_scores):,.2f}")
print(f"ORTALAMA TRAIN RMSE : {np.mean(train_rmse_scores):,.2f}")
print(f"ORTALAMA TEST RMSE  : {np.mean(val_rmse_scores):,.2f}")

# submission dosyasi 
submission = pd.DataFrame({
    "ID": test_ids,
    "Predicted_Production": test_predictions 
})

submission.to_csv("submission_xgb_final.csv", index=False)