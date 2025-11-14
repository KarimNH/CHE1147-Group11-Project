import pandas as pd
import numpy as np
import shap
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def explanation_shap(model, x_train, x_test, task_type, scaler = None, full_columns = None, index = 10, show_plot = False):
    """
    Generate SHAP explanations for a given model and dataset.
    Global explanations show feature importance across all test samples.
    Local explanations focus on a single instance (specified by `index`).

    Parameters:
    -----------
    model : object
        Trained machine learning model compatible with SHAP.
    x_train : pd.DataFrame or np.array
        Training feature data used to fit the model.
    x_test : pd.DataFrame or np.array
        Test feature data to explain predictions on.
    task_type : str
        Type of SHAP explanation:
        'global' for overall feature importance,
        'local' for individual prediction explanation.
    scaler : object, optional
        Pre-fitted scaler (e.g., StandardScaler) to transform x_train and x_test
    full_columns : list, optional
        List of column names for the dataset. Required only if a scaler is used
        (to retain proper column names after scaling).
    index : int, optional
        For local task: index of the sample to explain (default is 10).
        For global task: maximum number of features to display in the beeswarm plot.
    show_plot : bool, optional
        Whether to display the SHAP plots (beeswarm for global, waterfall for local).

    Returns:
    --------
    explainer : shap.Explainer
        SHAP explainer object fitted on the training data.
    shap_values : shap.Explanation
        SHAP values for the test dataset.
    shap_df : pd.DataFrame
        DataFrame containing SHAP values; either global (all samples) or local (single sample).
    """
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
        all_shap_values_array = shap_values.values
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
                random_state = 10,
                scoring=None,
                show_plot = False,
                plot_top_n = 10):
    """
    Calculate permutation feature importance for a model.

    Parameters
    ----------
    model : object
        Trained machine learning model compatible with permutation_importance.
    x_test : pd.DataFrame or np.array
        Test features to evaluate permutation importance.
    y_test : pd.Series or np.array
        True labels corresponding to x_test.
    scaler : object, optional
        Pre-fitted scaler (e.g., StandardScaler) to transform x_train and x_test
    full_columns : list, optional
        List of column names for the dataset. Required only if a scaler is used
        (to retain proper column names after scaling).
    n_repeats : int, optional
        Number of times to permute a feature (default is 10).
    random_state : int, optional
        Random seed for reproducibility (default is 10).
    scoring : str or callable, optional
        Scoring metric to evaluate model performance. If None, uses model's default score.
    show_plot : bool, optional
        Whether to display a horizontal bar plot of the top features (default is False).
        The plot shows mean permutation importance with error bars representing standard deviation.
    plot_top_n : int, optional
        Number of top features to display in the plot (default is 10).

    Returns
    -------
    result : sklearn.utils._permutation_importance.PermutationImportanceResult
        Object returned by sklearn's permutation_importance containing importances.
    importance_df : pd.DataFrame
        DataFrame containing feature names, mean importance, and standard deviation of importance.
    """
    if scaler is not None:
        x_test_scaled = scaler.transform(x_test)
        x_test = pd.DataFrame(x_test_scaled, columns=full_columns)
    else:
        x_test = x_test
    
    result = permutation_importance(model,
                                    x_test,
                                    y_test,
                                    n_repeats = n_repeats,
                                    random_state = random_state,
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
