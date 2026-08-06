import matplotlib.pyplot as plt

def scatter_plot(x,y,feature):
    plt.figure(figsize=(10,5))
    plt.scatter(x[feature],y)
    plt.xlabel(feature)
    plt.ylabel("disease pregression")
    plt.title(f"{feature} vs disease progression")
    plt.show()

def regression_line(x,y,prediction):
    plt.figure(figsize=(8,5))
    plt.scatter(x,y,label="Actual data")
    sorted_data=sorted(zip(x,prediction))

    x_sorted=[x for x,_ in sorted_data]
    y_sorted=[y for _,y in sorted_data]

    plt.plot(x_sorted,y_sorted,color="red",label="Regression line")

    plt.xlabel("BMI")
    plt.ylabel("Disease progression")
    plt.legend()
    plt.show()
