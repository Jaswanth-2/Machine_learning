import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.001,n_iteration=1000):
        self.learning_rate=learning_rate
        self.n_iteration=n_iteration
        self.weights=None
        self.bias=None

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def fit(self,x,y):
        n_samples,n_features=x.shape

        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iteration):
            linear_model=np.dot(x,self.weights)+self.bias
            y_predicted=self.sigmoid(linear_model)

            dw=(1/n_samples) * np.dot(x.T,(y_predicted-y))
            db=(1/n_samples) * np.sum(y_predicted-y)

            self.weights =self.weights - self.learning_rate * dw
            self.bias = self.bias -self.learning_rate * db

    def predict(self,x):
        linear_model=np.dot(x,self.weights)+self.bias
        y_predicted=self.sigmoid(linear_model)
        y_predicted_cls=[1 if i> 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

