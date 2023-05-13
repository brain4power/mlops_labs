import numpy as np
import pickle
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

if __name__ == '__main__':
    model = pickle.load(open('./models/lin_reg.pkl', 'rb'))

    x_test = np.load('./test/x_test_scaled.npy')
    y_test = np.load('./test/y_train.npy')

    y_pred = model.predict(x_test)

    print('Ошибка на тестовых данных')
    print('MSE: %.1f' % mse(y_test, y_pred))
    print('RMSE: %.1f' % mse(y_test, y_pred, squared=False))
    print('R2: %.4f' % r2_score(y_test, y_pred))
