import numpy as np


def generate_random_dataset(n:int,seed:int=None):
    """
    Function to generate a random dataset.
    Args:
    --------------------
    n(Inteager): number of values
    seed(Inteager): seed value for reproducibility of the data. Defaults to None
    Returns:
    ---------------------
    x(Numpy array): Input features.
    y(Numpy array): Output labels.
    """

    np.random.seed(seed)
    x = 2*np.random.rand(n,1)
    y = 4 + 3 * x + np.random.randn(n,1)

    return x,y

def linear_regression_model(x:float,w:float,b:float)->float:
    """Function to to generate linear regression predictions.
    The formula for the model function is : f(x) = wx+b. 
    Where:
    ----------------------
    w: weight/ slope.
    b: bias/ intercept.
    x: input features.
    Returns:
    -----------------------
    f(x): predicted values from the input, w and b."""

    return w * x +  b

def mean_squared_error(y_true, y_pred)->float:
    """
    Function to generate the mean squared error.
    So, from Andrew Ng's courses the cost-function is given to be:
    J(w,b) = 1/2m (Sum(y_pred-y_true)**2)
    
    Where:
    ----------------------
    J(w,b) - > cost function for linear regression / mean squared error.
    m -> total number of observations in the test set.
    
    Args:
    -----------------------
    y_true(array): true outputs.
    y_pred(array): predicted labels.
    Returns:
    -------------------------
    loss: mean cost value of the predictions."""
    loss = np.mean((y_true - y_pred))/2
    return loss

def gradtient_descent(x, y , w, b, learning_rate, epochs):
    m=len(x)
    costs = []

    for epoch in range(epochs):
        y_pred = linear_regression_model(x, w, b)
        cost = mean_squared_error(y, y_pred)
        costs.append(cost)

        #Compute gradients
        dw = -(2/m) * np.sum(x* (y - y_pred))
        db = -(2/m) * np.sum(y - y_pred)

        #Update weights and biases
        w-= learning_rate *dw
        b -= learning_rate * db

        if epoch %100 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")
    
    return w,b, costs

