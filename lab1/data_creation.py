import numpy as np


if __name__ == '__main__':
    def true_fun(x, a=np.pi, b=0, f=np.sin):
        x = np.atleast_1d(x)[:]
        a = np.atleast_1d(a)
        if f is None:
            f = lambda x: x
        x = np.sum([ai*np.power(x, i+1) for i, ai in enumerate(a)],
                   axis=0)
        return f(x+b)


    def noises(shape, noise_power):
        return np.random.randn(*shape) * noise_power


    def dataset(a, b, f=None, N=250, x_max=1, noise_power=0, random_x=True, seed=1234):
        np.random.seed(seed)
        if random_x:
            x = np.sort(np.random.rand(N))*x_max
        else:
            x = np.linspace(0, x_max, N)
        y_true = np.array([])
        for f_ in np.append([], f):
            y_true = np.append(y_true,
                               true_fun(x, a, b, f_))
        y_true = y_true.reshape(-1, N).T
        y = y_true + noises(y_true.shape, noise_power)
        return y, y_true, np.atleast_2d(x).T


    def train_test_split(x, y, train_size=None, test_size=None, random_state=42, shuffle=True, ):
        if random_state:
            np.random.seed(random_state)

        size = y.shape[0]
        idxs = np.arange(size)
        if shuffle:
            np.random.shuffle(idxs)

        if test_size and train_size is None:
            if test_size <= 1:
                train_size = 1 - test_size
            else:
                train_size = size - test_size
            test_size = None

        if train_size is None or train_size > size:
            train_size = size

        if train_size <= 1:
            train_size *= size

        if test_size is not None:
            if test_size <= 1:
                test_size *= size
            if test_size > size:
                test_size = size - train_size
        else:
            test_size = 0

        x_train, y_train = x[idxs[:int(train_size)]], y[idxs[:int(train_size)]]
        x_val, y_val = x[idxs[int(train_size):size - int(test_size)]],\
            y[idxs[int(train_size):size - int(test_size)]]

        if test_size > 0:
            x_test, y_test = x[idxs[size - int(test_size):]],\
                y[idxs[size - int(test_size):]]
            return x_train, y_train.squeeze(), x_val, y_val.squeeze(), x_test, y_test.squeeze()
        return x_train, y_train.squeeze(), x_val, y_val.squeeze()

    y, y_true, x = dataset(
        a=-200,
        b=200,
        f=None,
        N=500,
        x_max=111.2,
        noise_power=0.3,
        seed=555
    )

    x_train, y_train, x_test, y_test = train_test_split(
        x, y,
        test_size=0.3
    )

    np.save('./train/x_train.npy', x_train)
    np.save('./train/y_train.npy', y_train)
    np.save('./test/x_test.npy', x_test)
    np.save('./test/y_train.npy', y_test)
