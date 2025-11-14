import pandas as pd
import numpy as np
import shap
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def explanation_shap(model, x_train, x_test, task_type, full_columns = None, scaler = None, index = 10, show_plot = False):
    if scaler is not None:
        x_train_scaled = scaler.transform(x_train)
        x_train = pd.DataFrame(x_train_scaled, columns=full_columns)
        x_test_scaled = scaler.transform(x_test)
        x_test = pd.DataFrame(x_test_scaled, columns=full_columns)
    else:
        x_train = x_train
        x_test = x_test
    
    explainer = shap.Explainer(model, x_train)
    shap_values = explainer(x_test)
    
    if task_type == 'global':
        if show_plot:
            shap.plots.beeswarm(shap_values, max_display=index)
        all_shap_values_array = shap_values.values  # shape: (num_samples, num_features)
        feature_names = x_test.columns if hasattr(x_test, 'columns') else [f"Feature_{i}" for i in range(all_shap_values_array.shape[1])]
        shap_df = pd.DataFrame(all_shap_values_array, columns=feature_names)
    elif task_type == 'local':
        if show_plot:
            shap.plots.waterfall(shap_values[index])
        local_shap_values = shap_values[index].values.reshape(1, -1)
        feature_names = x_test.columns if hasattr(x_test, 'columns') else [f"Feature_{i}" for i in range(local_shap_values.shape[1])]
        shap_df = pd.DataFrame(local_shap_values, columns=feature_names)
    else:
        print("Invalid task_type. Use 'global' or 'local'.")
        shap_df = None
    return explainer, shap_values, shap_df

def per_imprt(model,
                x_test,
                y_test,
                scaler = None,
                full_columns = None,
                n_repeats = 10,
                random_state = 0,
                scoring=None,
                show_plot = False,
                plot_top_n = 10):
    """Calculate permutation importance and return the result."""
    if scaler is not None:
        x_train_scaled = scaler.transform(x_train)
        x_train = pd.DataFrame(x_train_scaled, columns=full_columns)
        x_test_scaled = scaler.transform(x_test)
        x_test = pd.DataFrame(x_test_scaled, columns=full_columns)
    else:
        x_train = x_train
        x_test = x_test
    
    result = permutation_importance(model,
                                    x_test,
                                    y_test,
                                    n_repeats=n_repeats,
                                    random_state=random_state,
                                    scoring=scoring)
    importance_df = pd.DataFrame({
        "feature": x_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    if show_plot == True:
        df_plot = importance_df.head(plot_top_n).copy()
        df_plot['importance_mean'] = pd.to_numeric(df_plot['importance_mean'], errors='coerce')
        df_plot['importance_std'] = pd.to_numeric(df_plot['importance_std'], errors='coerce').fillna(0)
        plt.figure(figsize=(10, 6))
        plt.barh(df_plot['feature'], df_plot['importance_mean'], xerr=df_plot['importance_std'], color='skyblue')
        plt.xlabel("Permutation Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance (Permutation)")
        plt.gca().invert_yaxis()
        plt.show()

    return result, importance_df 
