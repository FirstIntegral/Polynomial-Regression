'''
Data Source:

Title: [UK Consumer Trends: 1997 - 2022, Quarterly]
Author: [Matarr Gaye]
URL: [https://www.kaggle.com/datasets/matarrgaye/uk-consumer-trends-current-price?select=02CN.csv]
License: [CC0: Public Domain]

read README.md for more details
'''

#Imports
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

df = pd.read_csv('dataset/02CN.csv')

# Dropping rows 1 and 2. Not relevant data.
df = df.drop(df.index[[0, 1]])

# Fixing some columns names.
df.columns = ['Time period', 'Alcoholic beverages and tobacco', 'Alcaholic beverages', 'Spirits', 'Wines, cider and perry', 'Beer', 'Tobacco', 'Narcotics']

print(df.dtypes)
# All dtypes are objects, let's change all to int64 except "Time period" to stay as object
for column_name in df.columns:
    if column_name == 'Time period':
        df[column_name] = df[column_name].astype(object)
    else:
        df[column_name] = df[column_name].astype('int64')

print(df.dtypes)
# All good.

# Let's create a function to compute quarter numbers. Will be using this as a feature in the model later on
def quarter_number(s):
    year, quarter = s.split(' ')
    year = int(year)
    quarter = int(quarter[-1])
    return (year - 1997) * 4 + quarter

df['Quarter number'] = df['Time period'].apply(quarter_number)

df.set_index('Quarter number', inplace=True)

df.describe()
# No missing values. Pretty clean. Let's straight jump into the model. Let's try to predict spendings on "Tobacco" based on quarter number and predict the next one (2023 Q1)

X = df.index.values.reshape(-1, 1)
Y = df['Tobacco']

plt.scatter(X,Y)
plt.xlabel('Quarter number')
plt.ylabel('Spending (£ millions)')
plt.show()
# Gonna use a polynomial and not a linear regression. Makes it more accurate as it's clear from the graph.

# Partitioning the dataframe into training and test groups
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=64)

# Creating a Polynomial of degree 2
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Bulding model and fitting the training data
model = LinearRegression()
model.fit(X_train_poly, Y_train)

# Start predictions and then later on evaluate against the real data (using mean squared)
Y_pred = model.predict(X_test_poly)

# Calculating mean squared error and mean error
mse = mean_squared_error(Y_test, Y_pred)
me = math.sqrt(mse)

# Now the actual prediction for the next quarter
next_quarter = X[-1] + 1
next_quarter_poly = poly_features.transform(next_quarter.reshape(-1, 1))
next_quarter_spending = model.predict(next_quarter_poly)[0]

# Calculating r-squared
r_squared = model.score(X_test_poly, Y_test)

# getting the model formula
def polynomial_formula(coefficients, degree):
    formula = "y = "
    for i in range(degree + 1):
        if i == 0:
            formula += f"{coefficients[i]:.3f}"
        else:
            formula += f" + {coefficients[i]:.3f} * x^{i}"
    return formula

coefficients = np.concatenate(([model.intercept_], model.coef_[1:]))
formula = polynomial_formula(coefficients, degree)

# Finally, plotting
plt.scatter(X, Y, color='blue', label='Actual data points')

# Plotting the polynomial regression model
X_plot = np.linspace(X.min(), X.max(), num=100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
plt.plot(X_plot, model.predict(X_plot_poly), color='red', label='Regression Curve')

# Plotting the test set predictions
plt.scatter(X_test, Y_pred, color='green', marker='x', label='Test Set Predictions')

plt.xlabel('Quarters number')
plt.ylabel('Spending (£ millions)')
plt.title('UK Consumer Trends (Tobacco) Polynomial Regression Model')

plt.legend()

print("Polynomial Regression Equation:", formula)
print("Mean Squared Error:", mse)
print("Mean Error:", me)
print("r-squared:", r_squared)
print("Predicted spending on tobacco for the next quarter:", next_quarter_spending)

plt.show()

