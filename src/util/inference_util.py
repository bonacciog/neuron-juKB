from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import numpy as np

def print_and_inference(predictions, int2label_dict):

    #print scores and final prediction
    print("\nAll scores:")
    for i, score in enumerate(predictions.tolist()):
        print(int2label_dict[i] + " - score: "+ str(round(score,3)))

    # numpy array
    if isinstance(predictions, (np.ndarray, np.generic) ):
        value = np.amax(predictions)
        index = np.argmax(predictions, axis=0)

    #torch Tensor
    elif torch.is_tensor(predictions):
        value, index = predictions.max(0)
    print("\nModel predicts \"" + int2label_dict[index.item()] + "\" with accuracy " + str(round(value.item() * 100 , 1)) + "%")

    return index.item()

def get_scores(predictions, int2label_dict):

    score_dict = {}
    #print scores and final prediction
    for i, score in enumerate(predictions.tolist()):
        label = int2label_dict[i]
        score_dict[label] = str(round(score,3))

    # numpy array
    if isinstance(predictions, (np.ndarray, np.generic) ):
        value = np.amax(predictions)
        index = np.argmax(predictions, axis=0)

    #torch Tensor
    elif torch.is_tensor(predictions):
        value, index = predictions.max(0)

    return index.item() , score_dict

def print_confusion_matrix(y_true, y_preds, labels, verbose=True):
    if verbose:
        index = ["true:"+l for l in labels]
        columns = ["pred:"+l for l in labels]
        cmtx = pd.DataFrame(
            confusion_matrix(y_true, y_preds, labels=labels), 
            index=index, 
            columns=columns
        )
        print(cmtx)
    else:
        print(confusion_matrix(y_true, y_preds, labels=labels))