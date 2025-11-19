from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, max_error, mean_absolute_percentage_error)

def get_regression_metrics(model, X, y_true):
    """
    Get a dicionary with regression metrics,
    including MAE, MSE, Max Error, MAPE, R2 Score
    """
    y_predicted = model.predict(X)

    mae = mean_absolute_error(y_true, y_predicted)
    mse = mean_squared_error(y_true, y_predicted)
    maximum_error = max_error(y_true, y_predicted)
    mape = mean_absolute_percentage_error(y_true, y_predicted)
    r2 = r2_score(y_true, y_predicted)

    metrics_dict = {
        'mae': mae,
        'mse': mse,
        'max_error': maximum_error,
        'mape': mape,
        'r2': r2
    }

    return metrics_dict
