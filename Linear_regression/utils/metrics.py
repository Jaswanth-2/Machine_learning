import math

def mean_absolute_error(y_true, y_pred):
    total_error = 0
    for actual, predicted in zip(y_true, y_pred):
        total_error += abs(actual - predicted)
    return total_error / len(y_true)

def mean_squared_error(y_true, y_pred):
    total_error = 0
    for actual, predicted in zip(y_true, y_pred):
        total_error += (actual - predicted) ** 2
    return total_error / len(y_true)

def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return math.sqrt(mse)

def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = 0
    ss_tot = 0
    for actual, predicted in zip(y_true, y_pred):

        ss_res += (actual - predicted) ** 2
        ss_tot += (actual - mean_y) ** 2

    return 1 - (ss_res / ss_tot)