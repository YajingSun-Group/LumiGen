import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

def compute_metrics(filename):
    df = pd.read_csv(filename)
    y = df['y_test_csv']
    output = df['test_output_csv']
    mae = mean_absolute_error(y, output)
    mse = mean_squared_error(y, output)
    r2 = r2_score(y, output)
    rmse = math.sqrt(mse)

    return mae, mse, r2, rmse

