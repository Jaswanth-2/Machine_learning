import numpy as np

def train_test_split(x,y,test_size=0.2,random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_samples=len(x)
    indices=np.arange(n_samples)
    np.random.shuffle(indices)
    test_count=int(n_samples* test_size)

    test_indices=indices[:test_count]
    train_indices=indices[test_count:]

    x_train=x.iloc[train_indices]
    x_test=x.iloc[test_indices]

    y_train=y.iloc[train_indices]
    y_test=y.iloc[test_indices]

    return x_train,x_test,y_train,y_test