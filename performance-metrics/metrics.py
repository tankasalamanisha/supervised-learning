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

def true_positive(y_true, y_pred):
    """Function to calculate True Positive
    Args:
    -----------------
    y_true: list of true/actual values.
    y_pred: list of predicted values.

    Returns:
    -----------------
    int: count of true positives.
    """
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp ==1:
            tp+=1
    return tp
def true_negative(y_true, y_pred):
    """Function to calculate true negatives.
    Args:
    -----------------
    y_true: list of true/actual values.
    y_pred: list of predicted values.

    Returns:
    -----------------
    int: count of true negative.
    """
    tn=0
    for yt,yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn+=1
    return tn
def false_positive(y_true, y_pred):
    """Function to calculate false positive.
    Args:
    -----------------
    y_true: list of true/actual values.
    y_pred: list of predicted values.

    Returns:
    -----------------
    int: count of false positives.
    """
    fp = 0
    for yt,yp in zip(y_true,y_pred):
        if yt == 0 and yp == 1:
            fp+=1
    return fp
def false_negative(y_true,y_pred):
    """Function to calculate false negative.
    Args:
    -----------------
    y_true: list of true/actual values.
    y_pred: list of predicted values.

    Returns:
    -----------------
    int: count of false negative.
    """
    fn = 0
    for yt,yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn+=1
    return fn

def accuracy2(y_true,y_pred):
    """Function to calculate accuracy using tp,tn,fp,fn.
    """
    tp = true_positive(y_true,y_pred)
    tn = true_negative(y_true,y_pred)
    fp = false_positive(y_true,y_pred)
    fn = false_negative(y_true,y_pred)

    accuracy_score = (tp+tn)/(tp+tn+fp+fn)
    return accuracy_score

def precision(y_true,y_pred):
    """Function to calculate precision from tp,fp.
    """
    tp=true_positive(y_true,y_pred)
    fp= false_positive(y_true,y_pred)

    precision= tp/(tp+fp)
    return precision

def recall(y_true, y_pred):
    """Function to calculate recall from tp,fn.
    """
    tp = true_positive(y_true,y_pred)
    fn = false_negative(y_true,y_pred)

    recall = tp/(tp+fn)
    return recall

def f1(y_true,y_pred):
    """
    Function to calculate f1 score.
    """
    p= precision(y_true,y_pred)
    r= recall(y_true,y_pred)

    score = 2*p*r/(p+r)

    return score

def tpr(y_true,y_pred):
    """
    Function to calculate tpr.
    TPR is also known as sensitivity and is similar to recall.
    """

    return recall(y_true,y_pred)

def fpr(y_true,y_pred):
    """
    Function to calculate fpr.
    FPR is also known as specificity.
    """
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)

    return fp/(tn+fp)
