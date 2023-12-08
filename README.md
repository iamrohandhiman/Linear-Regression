# Data Preparation

## Description:
This code cell initializes training data for a simple machine learning model. It creates two NumPy arrays, `x_train` and `y_train`, to be used as features and corresponding labels, respectively.

## Variables:
- `x_train`: NumPy array containing the feature values `[1, 2, 3]`.
- `y_train`: NumPy array containing the corresponding label values `[500.0, 750.0, 1000.0]`.

## Usage:
This data can be used to train a regression model where `x_train` represents the input features, and `y_train` represents the corresponding output labels.


# Computing Cost Function

## Description:
This code cell defines a function `computing_cost` to calculate the cost of a linear regression model. The cost is computed using the mean squared error between the predicted values and the actual labels.

## Parameters:
- `x`: NumPy array representing the feature values (numpy.ndarray, shape: (m, ) where m is the number of samples).
- `y`: NumPy array representing the corresponding label values (numpy.ndarray, shape: (m, ) where m is the number of samples).
- `w`: Weight parameter of the linear regression model (float).
- `b`: Bias parameter of the linear regression model (float).

## Returns:
- `total_cost`: The computed cost using the mean squared error formula (float).

## Implementation Details:
The cost function (mean squared error) is given by the formula:

$$ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (w \cdot x_i + b - y_i)^2 $$

where:
- \( J(w,b) \) is the cost function.
- \( m \) is the number of samples.
- \( w \) is the weight parameter.
- \( b \) is the bias parameter.
- \( x_i \) is the ith feature.
- \( y_i \) is the ith label.

The function iterates through each data point, computes the predicted value using the linear regression model, calculates the squared difference between the predicted and actual values, and accumulates these squared differences. Finally, it divides the total by twice the number of data points (\(2m\)) to obtain the mean squared error.

# Vectorized Computing Cost Function

## Description:
This code cell defines a vectorized version of the `computing_cost` function, referred to as `computing_cost_vectorised`. The vectorized implementation takes advantage of NumPy operations to improve efficiency.

## Parameters:
- `x`: NumPy array representing the feature values (numpy.ndarray, shape: (m, ) where m is the number of samples).
- `y`: NumPy array representing the corresponding label values (numpy.ndarray, shape: (m, ) where m is the number of samples).
- `w`: Weight parameter of the linear regression model (float).
- `b`: Bias parameter of the linear regression model (float).

## Returns:
- `total_cost`: The computed cost using the mean squared error formula (float).

## Vectorized Implementation:
The cost function (mean squared error) is computed using vectorized operations in NumPy:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (w \cdot x_i + b - y_i)^2 $$

The vectorized version simplifies the computation by removing the need for explicit loops. It takes advantage of NumPy's ability to perform element-wise operations on entire arrays.


## Documentation for `compute_cost_vectorized` Function

## Description
The `compute_cost_vectorized` function calculates the cost (or loss) of a linear regression model using vectorized operations.

## Parameters
- `x` (numpy array): Input features with shape (n, m), where n is the number of examples and m is the number of features.
- `y` (numpy array): True labels with shape (n, 1), where n is the number of examples.
- `w` (numpy array): Weight vector with shape (m, 1), representing the weights for each feature.
- `b` (float): Bias term.

## Formula
The cost is computed using the mean squared error (MSE) formula:

$$ J(w, b) = \frac{1}{2n} \sum_{i=1}^{n} (w \cdot x_i + b - y_i)^2 $$

Where:
- \( n \) is the number of examples.
- \( x_i \) is the feature vector for the \(i^{th}\) example.
- \( y_i \) is the true label for the \(i^{th}\) example.
- \( w \) is the weight vector.
- \( b \) is the bias term.

## Returns
- `total_cost` (float): The computed cost using the given input data and model parameters.

## Example
```python
import numpy as np

# Example data
x = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([[5], [8], [11]])
w = np.array([[2], [1]])
b = 1.5

# Calculate the cost
cost = compute_cost_vectorized(x, y, w, b)
print("Cost:", cost)
```

In this example, `x` is a matrix of input features, `y` is a column vector of true labels, `w` is a column vector of weights, and `b` is the bias term. The function calculates the cost based on the provided data and model parameters.

This vectorized implementation efficiently computes the cost for the entire dataset in a single operation, improving computational performance.

# Documentation for Gradient Descent Optimization

## Description
The provided code implements gradient descent optimization to find the optimal parameters (weights `w` and bias `b`) for a linear regression model. The optimization process aims to minimize the cost function by iteratively updating the parameters.

## Parameters
- `x_train` (numpy array): Input features for training with shape (n, m), where n is the number of examples and m is the number of features.
- `y_train` (numpy array): True labels for training with shape (n, 1), where n is the number of examples.
- `w` (float): Initial weight parameter.
- `b` (float): Initial bias parameter.
- `alpha` (float): Learning rate, controlling the step size in each iteration.

## Variables
- `prev_cost` (float): Cost from the previous iteration.
- `current_cost` (float): Cost from the current iteration.
- `count` (int): Iteration counter.
- `w_history` (list): List to store the history of weight values during optimization.
- `b_history` (list): List to store the history of bias values during optimization.

## Optimization Loop
The optimization loop continues until the absolute difference between the current and previous costs is less than \(1 \times 10^{-5}\).

## Gradient Descent Update Formulas

The weight parameter \(w\) is updated using the following formula:

$$ w = w - \alpha \cdot \frac{\sum_{i=1}^{n} (w \cdot x_i + b - y_i) \cdot x_i}{n} $$

Where:
- \( \alpha \) is the learning rate.
- \( n \) is the number of examples.
- \( x_i \) is the feature vector for the \(i^{th}\) example.
- \( y_i \) is the true label for the \(i^{th}\) example.

The bias term \(b\) is updated using the formula:

$$ b = b - \alpha \cdot \frac{\sum_{i=1}^{n} (w \cdot x_i + b - y_i)}{n} $$

Where:
- \( alpha \) is the learning rate.
- \( n \) is the number of examples.
- \( x_i \) is the feature vector for the \(i^{th}\) example.
- \( y_i \) is the true label for the \(i^{th}\) example.
- 
```python
while abs(current_cost - prev_cost) > 1e-5:
    count = count + 1
    prev_cost = current_cost
   
    current_cost = compute_cost_vectorized(x_train, y_train, w, b)
    
    w_history.append(w)
    b_history.append(b)

    if(current_cost <= 1e-5):
        break

    w = w - alpha * ((np.sum((w * x_train + b - y_train) * x_train)) / x_train.shape[0])
    b = b - alpha * ((np.sum(w * x_train + b - y_train)) / x_train.shape[0])
```

## Output
- `w` (float): Optimal weight parameter.
- `b` (float): Optimal bias parameter.
- `count` (int): Number of iterations performed.
- `current_cost` (float): Final cost after optimization.

## Example Usage
```python
# Initial parameters
prev_cost = 0
current_cost = 10
count = 0
b = -10
w = 30
alpha = 1 

# Lists to store parameter history
w_history = []
b_history = []

# Run optimization
while abs(current_cost - prev_cost) > 1e-5:
    count = count + 1
    prev_cost = current_cost
   
    current_cost = compute_cost_vectorized(x_train, y_train, w, b)
    
    w_history.append(w)
    b_history.append(b)

    if(current_cost <= 1e-5):
        break

    w = w - alpha * ((np.sum((w * x_train + b - y_train) * x_train)) / x_train.shape[0])
    b = b - alpha * ((np.sum(w * x_train + b - y_train)) / x_train.shape[0])

# Print results
print("Optimal weight:", w)
print("Optimal bias:", b)
print("Number of iterations:", count)
print("Final cost:", current_cost)
```

In this example, the code initializes parameters, runs the optimization loop, and prints the optimal weight, bias, number of iterations, and final cost after optimization. You can use this template for your specific linear regression problem.

# Analysis of Cost Function and Gradient Descent

## Cost Function Analysis

### Cost Function Formula
The cost function, denoted as \(J(w, b)\), measures the difference between the predicted values and the actual labels. In the context of linear regression, it is commonly defined as the mean squared error (MSE):

$$ J(w, b) = \frac{1}{2n} \sum_{i=1}^{n} (w \cdot x_i + b - y_i)^2 $$

Where:
- \( n \) is the number of examples.
- \( x_i \) is the feature vector for the \(i^{th}\) example.
- \( y_i \) is the true label for the \(i^{th}\) example.
- \( w \) is the weight vector.
- \( b \) is the bias term.

### Cost Function Visualization
The code snippet provided generates a plot of the cost function with respect to different values of the weight parameter \(w\). This visualization helps us understand how the cost changes as the weight varies.

```python
w_values = np.linspace(-500, 500, 1000)

cost_values = np.zeros_like(w_values)

for i, w_test in enumerate(w_values):
    cost_values[i] = compute_cost_vectorized(x_train, y_train, w_test, b)

plt.figure(figsize=(10, 6))
plt.plot(w_values, cost_values, label='Cost Function')
plt.scatter(w, current_cost, color='red', marker='




