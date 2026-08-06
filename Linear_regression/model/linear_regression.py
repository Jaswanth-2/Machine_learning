class LinearRegression:

    def __init__(self):
        self.slope=0
        self.intercept=0
    
    def fit(self,x,y):
        mean_x=sum(x)/len(x)
        mean_y=sum(y)/len(y)
        print("Mean X:",mean_x)
        print("Mean_y :",mean_y)

        varience=0
        for value in x:
            varience += (value - mean_x) **2
        print("Vrience", varience)

        covariance =0
        for x_value,y_value in zip(x,y):
            covariance += (x_value - mean_x)*(y_value - mean_x)
        print("Covarience",covariance)

        self.slope=covariance/varience
        self.intercept=mean_y - self.slope * mean_x

    def predict(self,x):
        prediction=[]

        for value in x:
            y_pred=self.slope * value +self.intercept
            prediction.append(y_pred)
        return prediction