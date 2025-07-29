import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay

# --- Preprocessing Function ---
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace(r"\s+", "", regex=True).str.upper()
    df.dropna(inplace=True)

    features = ['MAKER', 'MODEL', 'VEHICLECLASS', 'ENGINESIZE',
                'CYLINDERS', 'TRANSMISSION', 'FUEL', 'FUELCONSUMPTION']
    target = 'COEMISSIONS'

    X = df[features]
    y = df[target]
    cat_features = ['MAKER', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUEL']
    return X, y, cat_features

# --- Evaluation Function ---
def evaluate_model(model, X_test, y_test, cat_features):
    test_pool = Pool(X_test, cat_features=cat_features)
    predictions = model.predict(test_pool)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nModel Performance Metrics:")
    print(f"MAE  : {mae:.2f} g/km")
    print(f"RMSE : {rmse:.2f} g/km")
    print(f"R²   : {r2:.4f}")

    # Save metrics to file
    with open("model_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Model Performance Metrics\n")
        f.write(f"MAE  : {mae:.2f} g/km\n")
        f.write(f"RMSE : {rmse:.2f} g/km\n")
        f.write(f"R²   : {r2:.4f}\n")

    # --- Binned Confusion Matrix ---
    y_true_bin = pd.cut(y_test, bins=[0, 113, float("inf")], labels=["Acceptable", "High"])
    y_pred_bin = pd.cut(predictions, bins=[0, 113, float("inf")], labels=["Acceptable", "High"])

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=["Acceptable", "High"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Acceptable", "High"])
    disp.plot(cmap="Blues")
    plt.title("Binned Confusion Matrix (Threshold: 113 g/km)")
    plt.savefig("binned_confusion_matrix.png")
    plt.show()

# --- Main Training Flow ---
def main():
    file_path = "co2_emission_combined.csv"
    X, y, cat_features = preprocess_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(verbose=0)
    param_grid = {
        'iterations': [300, 500],
        'learning_rate': [0.05, 0.1],
        'depth': [4, 6, 8]
    }

    # --- Hyperparameter tuning ---
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train, cat_features=cat_features)

    best_model = grid.best_estimator_
    best_model.save_model("catboost_co2_model.cbm")
    print("\n Best model trained and saved as 'catboost_co2_model.cbm'")
    print("Best Hyperparameters:", grid.best_params_)

    evaluate_model(best_model, X_test, y_test, cat_features)

if __name__ == "__main__":
    main()
