import time
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(int(time.time()))

n_scores_per_test = 50000
test_ratios = [.1, .2, .4, .6, .8]
model = Lasso(alpha=0.0005)

# load our training data
train = pd.read_csv('train.csv')
X = train[['OverallCond']].values
y = np.log(train['SalePrice'].values)

# custom RMSE
def rmse(y_predicted, y_actual):
    tmp = np.power(y_actual - y_predicted, 2) / y_actual.size
    return np.sqrt(np.sum(tmp, axis=0))

for test_ratio in test_ratios:
    print 'Testing test ratio:', test_ratio

    scores = []
    avg_scores = []
    for i in range(n_scores_per_test):
        if i % 200 == 0:
            print i, '/', n_scores_per_test

        seed = np.random.randint(2**32-1)
        X, y = shuffle(X, y, random_state=seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores.append(rmse(y_pred, y_test))
        avg_scores.append(np.array(scores).mean())

    plt.plot(avg_scores, label=str(test_ratio))

plt.legend(loc='upper right')
plt.show()