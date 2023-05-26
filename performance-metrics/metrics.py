import pandas as pd
import numpy as np
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

def auc_roc_curve(y_true:list,y_pred:list, thresholds:list):
    # empty lists to store tpr and fpr values
    tpr_list =[]
    fpr_list =[]
    for thresh in thresholds:
        temp_pred = [1 if x>=thresh else 0 for x in y_pred]
        # calculate tpr
        temp_tpr = tpr(y_true, temp_pred)
        # calculate fpr
        temp_fpr = fpr(y_true, temp_pred)

        # appending to our lists:
        tpr_list.append(temp_tpr)
        fpr_list.append(temp_fpr)

    tpr_fpr_df = pd.DataFrame(data=dict(zip(['threshold','tpr','fpr'],[thresholds, tpr_list, fpr_list])))
    
    return tpr_fpr_df

def log_loss(y_true:list,y_pred:list)->np.float:
    """Function to calculate logloss"""
    epsilon = 1e-15

    loss =[]
    for yt,yp in zip(y_true,y_pred):

        yp = np.clip(yp, epsilon, 1-epsilon)

        temp_loss = -1.0 * (
            yt * np.log(yp)
            + (1-yt)*np.log(1-yp)
        )
        loss.append(temp_loss)
    
    return np.mean(loss)