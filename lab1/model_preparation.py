import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

if __name__ == '__main__':
    x_train = np.load('./train/x_train_scaled.npy')
    y_train = np.load('./train/y_train.npy')

    model = LinearRegression(
        fit_intercept=True,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    pickle.dump(model, open('./models/lin_reg.pkl', 'wb'))
