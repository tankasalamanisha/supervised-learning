def accuracy(y_true, y_pred):
    """Function to compute accuracy.
    Args:
    ---------------
    y_true: list of true/actual values.
    y_pred: list of predicted values.
    
    Returns:
    accuracy_score(float): accuracy of the predictions made."""
    
    #flag for correct predictions
    flag = 0
    # looping through y_true and y_pred

    for yt,yp in zip(y_true,y_pred):
        if yt == yp:
            flag +=1

    # Return Accuracy
    return flag/len(y_true)
