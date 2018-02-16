# RMSE Convergence

Plots the average the RMSE error over time given different train/test ratio sizes:

![](https://raw.github.com/benji/rmse_convergence/master/rmse_average_over_time_zoomed.png)

![](https://raw.github.com/benji/rmse_convergence/master/rmse_average_over_time.png)

Data has been taken from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

----
### RMSE score

    def rmse(y_predicted, y_actual):
      tmp = np.power(y_actual - y_predicted, 2) / y_actual.size
      return np.sqrt(np.sum(tmp, axis=0))

