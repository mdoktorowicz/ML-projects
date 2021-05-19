from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather Data

boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(boston_dataset.target)
log_prices.shape
target = pd.DataFrame(log_prices, columns=['PRICE'])
target.shape

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

property_stats = features.mean().values.reshape(1, 11)
property_stats


regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

mse = mean_squared_error(target, fitted_vals)
# print(f"MSE is: {round(mse, 3)}")
RMSE = np.sqrt(mse)
RMSE
# print(f"RMSE is: {round(np.sqrt(mse), 3)}")
# omitted_normal_rsquared = round(results.rsquared, 3)

# Inflation multiplier = median home price today / median home price in dataset year
inflation_multiplier = 666/np.median(boston_dataset.target)
inflation_multiplier


def get_log_estimate(nr_rooms, students_per_classroom, next_to_river=False, high_confidence=True):
    '''
    Estimates house price in Boston, based on input arguments.
    Price adjusted for inflation.

    Keyword arguments:
    rm -- number of rooms in property
    students_per_classroom -- number of students per teacher at schools in the area
    next_to_river -- True if property next to river, False otherwise
    high_confidence -- 95% confidence rate if True, False otherwise

    '''

    if nr_rooms < 1 or students_per_classroom < 1:
        print('Parameters unrealistic. Try again.')
        return

    # Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom

    if next_to_river:
        property_stats[0][CHAS_IDX] = 1

    else:
        property_stats[0][CHAS_IDX] = 0

    # Make prediction
    log_estimate = regr.predict(property_stats)[0][0]
    estimate = np.e ** log_estimate
    estimate = estimate * inflation_multiplier

    # Calc range
    if high_confidence:
        upper_bound = log_estimate + 2 * np.e ** RMSE * inflation_multiplier
        lower_bound = log_estimate - 2 * np.e ** RMSE * inflation_multiplier
        interval = 95
    else:
        upper_bound = log_estimate + 1 * np.e ** RMSE * inflation_multiplier
        lower_bound = log_estimate - 1 * np.e ** RMSE * inflation_multiplier
        interval = 68

    print(f'Estimated house price is {int(estimate * 1000)} USD.')
    print(f'At {interval}%, the valuation range is: ')
    print(
        f'{int((estimate - lower_bound) * 1000)} USD at the low end, {int((estimate - upper_bound) * 1000)} USD at the high end')

