import numpy as np


class KNN:

    def __init__(self, k=5):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance

    def predict_one(self, x):
        distances = []

        for i in range(len(self.x_train)):
            dist = self.euclidean_distance(x, self.x_train[i])
            distances.append((dist, self.y_train.iloc[i]))

        distances.sort(key=lambda x: x[0])

        k_nearest = distances[:self.k]

        labels = [label for distance, label in k_nearest]

        prediction = max(set(labels), key=labels.count)

        return prediction

    def predict(self, X):
        predictions = []

        for x in X:
            prediction = self.predict_one(x)
            predictions.append(prediction)

        return np.array(predictions)