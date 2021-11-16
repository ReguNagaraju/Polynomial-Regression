# Polynomial-Regression

Introduction to Polynomial Regression (with Python Implementation)

Here’s Everything you Need to Get Started with Polynomial Regression
What’s the first machine learning algorithm you remember learning? The answer is typically linear regression for most of us (including myself). Honestly, linear regression props up our machine learning algorithms ladder as the basic and core algorithm in our skillset.

But what if your linear regression model cannot model the relationship between the target variable and the predictor variable? In other words, what if they don’t have a linear relationship?

![image](https://user-images.githubusercontent.com/92477493/141952950-a88ed14f-c487-4265-9fed-f979c4606899.png)

Well – that’s where Polynomial Regression might be of assistance. In this article, we will learn about polynomial regression, and implement a polynomial regression model using Python.

If you are not familiar with the concepts of Linear Regression, then I highly recommend you read this article before proceeding further.

Let’s dive in!

What is Polynomial Regression?
Polynomial regression is a special case of linear regression where we fit a polynomial equation on the data with a curvilinear relationship between the target variable and the independent variables.

In a curvilinear relationship, the value of the target variable changes in a non-uniform manner with respect to the predictor (s).

In Linear Regression, with a single predictor, we have the following equation:

linear regression equation

where,

          Y is the target,

          x is the predictor,

          𝜃0 is the bias,

          and 𝜃1 is the weight in the regression equation

This linear equation can be used to represent a linear relationship. But, in polynomial regression, we have a polynomial equation of degree n represented as:

polynomial regression equation

Here:

          𝜃0 is the bias,

          𝜃1, 𝜃2, …, 𝜃n are the weights in the equation of the polynomial regression,

          and n is the degree of the polynomial

The number of higher-order terms increases with the increasing value of n, and hence the equation becomes more complicated.

 

Polynomial Regression vs. Linear Regression
Now that we have a basic understanding of what Polynomial Regression is, let’s open up our Python IDE and implement polynomial regression.

I’m going to take a slightly different approach here. We will implement both the polynomial regression as well as linear regression algorithms on a simple dataset where we have a curvilinear relationship between the target and predictor. Finally, we will compare the results to understand the difference between the two.

First, import the required libraries and plot the relationship between the target variable and the independent variable:

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for calculating mean_squared error
from sklearn.metrics import mean_squared_error

# creating a dataset with curvilinear relationship
x=10*np.random.normal(0,1,70)
y=10*(-x**2)+np.random.normal(-100,100,70)

# plotting dataset
plt.figure(figsize=(10,5))
plt.scatter(x,y,s=15)
plt.xlabel('Predictor',fontsize=16)
plt.ylabel('Target',fontsize=16)
plt.show()

![image](https://user-images.githubusercontent.com/92477493/141953108-fe8ed9a7-ec21-4f76-8415-8c3d50067e89.png)

Let’s start with Linear Regression first:

# Importing Linear Regression
from sklearn.linear_model import LinearRegression

# Training Model
lm=LinearRegression()
lm.fit(x.reshape(-1,1),y.reshape(-1,1))
Let’s see how linear regression performs on this dataset:

y_pred=lm.predict(x.reshape(-1,1))

# plotting predictions
plt.figure(figsize=(10,5))
plt.scatter(x,y,s=15)
plt.plot(x,y_pred,color='r')
plt.xlabel('Predictor',fontsize=16)
plt.ylabel('Target',fontsize=16)
plt.show()

![image](https://user-images.githubusercontent.com/92477493/141953170-19e7439b-a8fe-40da-9ed9-0458fdc1eb8c.png)

print('RMSE for Linear Regression=>',np.sqrt(mean_squared_error(y,y_pred)))
rmse linear regression

Here, you can see that the linear regression model is not able to fit the data properly and the RMSE (Root Mean Squared Error) is also very high.

Now, let’s try polynomial regression.
The implementation of polynomial regression is a two-step process. First, we transform our data into a polynomial using the PolynomialFeatures function from sklearn and then use linear regression to fit the parameters:

![image](https://user-images.githubusercontent.com/92477493/141953250-b329632b-70fe-4e54-a0fe-c7c0e81c3d8d.png)

We can automate this process using pipelines. Pipelines can be created using Pipeline from sklearn.

Let’s create a pipeline for performing polynomial regression:

# importing libraries for polynomial transform
from sklearn.preprocessing import PolynomialFeatures
# for creating pipeline
from sklearn.pipeline import Pipeline
# creating pipeline and fitting it on data
Input=[('polynomial',PolynomialFeatures(degree=2)),('modal',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(x.reshape(-1,1),y.reshape(-1,1))
Here, I have taken a 2-degree polynomial. We can choose the degree of polynomial based on the relationship between target and predictor. The 1-degree polynomial is a simple linear regression; therefore, the value of degree must be greater than 1.

With the increasing degree of the polynomial, the complexity of the model also increases. Therefore, the value of n must be chosen precisely. If this value is low, then the model won’t be able to fit the data properly and if high, the model will overfit the data easily.

Read more about underfitting and overfitting in machine learning here.

Let’s take a look at our model’s performance:

poly_pred=pipe.predict(x.reshape(-1,1))
#sorting predicted values with respect to predictor
sorted_zip = sorted(zip(x,poly_pred))
x_poly, poly_pred = zip(*sorted_zip)
#plotting predictions
plt.figure(figsize=(10,6))
plt.scatter(x,y,s=15)
plt.plot(x,y_pred,color='r',label='Linear Regression')
plt.plot(x_poly,poly_pred,color='g',label='Polynomial Regression')
plt.xlabel('Predictor',fontsize=16)
plt.ylabel('Target',fontsize=16)
plt.legend()
plt.show()

![image](https://user-images.githubusercontent.com/92477493/141953398-284f9d8e-2fd9-4d0e-a312-fbb8b00385cf.png)

print('RMSE for Polynomial Regression=>',np.sqrt(mean_squared_error(y,poly_pred)))
rmse polynomial regression

We can clearly observe that Polynomial Regression is better at fitting the data than linear regression. Also, due to better-fitting, the RMSE of Polynomial Regression is way lower than that of Linear Regression.

But what if we have more than one predictor?
For 2 predictors, the equation of the polynomial regression becomes:

two degree polynomial regression

where,

          Y is the target,

          x1, x2 are the predictors,

          𝜃0 is the bias,

          and, 𝜃1, 𝜃2, 𝜃3, 𝜃4, and 𝜃5 are the weights in the regression equation

For n predictors, the equation includes all the possible combinations of different order polynomials. This is known as Multi-dimensional Polynomial Regression.

But, there is a major issue with multi-dimensional Polynomial Regression – multicollinearity. Multicollinearity is the interdependence between the predictors in a multiple dimensional regression problem. This restricts the model from fitting properly on the dataset.
