from dataset.data import load_data
from utils.visualization import scatter_plot, regression_line
from utils.split import train_test_split
from utils.metrics import (mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score)
from model.linear_regression import LinearRegression

x, y = load_data()

print("Dataset Loaded Successfully")
print("Feature Shape :", x.shape)
print("Target Shape  :", y.shape)

scatter_plot(x, y, "bmi")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=40
)

print("\nAfter Train-Test Split")
print("X Train :", x_train.shape)
print("X Test  :", x_test.shape)
print("Y Train :", y_train.shape)
print("Y Test  :", y_test.shape)

model = LinearRegression()
model.fit(x_train["bmi"], y_train)

print("\nModel Parameters")
print("Slope     :", model.slope)
print("Intercept :", model.intercept)

predictions = model.predict(x_test["bmi"])

print("\nFirst 10 Predictions")
for actual, predicted in zip(y_test[:10], predictions[:10]):
    print(f"Actual: {actual:6.1f} | Predicted: {predicted:8.2f}")

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("-" * 30)
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

regression_line(
    x_test["bmi"],
    y_test,
    predictions
)