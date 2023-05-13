import numpy as np
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    x_train = np.load('./train/x_train.npy')
    y_train = np.load('./train/y_train.npy')
    x_test = np.load('./test/x_test.npy')
    y_test = np.load('./test/y_train.npy')

    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    np.save('./train/x_train_scaled.npy', x_train_scaled)
    np.save('./test/x_test_scaled.npy', x_test_scaled)
