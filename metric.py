from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
import copy


import math

def metric_mae(y_pred, y_true):
    perc_y_pred = y_pred.detach().cpu().numpy()
    perc_y_true = y_true.detach().cpu().numpy()
    mae = mean_absolute_error(perc_y_true, perc_y_pred, multioutput='raw_values')[0]
    return mae


def metric_rmse(y_pred, y_true):
    perc_y_pred = y_pred.detach().cpu().numpy()
    perc_y_true = y_true.detach().cpu().numpy()
    mse = mean_squared_error(perc_y_true, perc_y_pred, multioutput='raw_values')[0]
    rmse = math.sqrt(mse)
    return rmse


def metric_mape(y_pred, y_true):
    perc_y_pred = y_pred.detach().cpu().numpy()
    perc_y_true = y_true.detach().cpu().numpy()
    mape = mean_absolute_percentage_error(perc_y_true, perc_y_pred, multioutput='raw_values')[0]
    return mape

## you should call detach before cpu to prevent superfluous gredient copying 
# detach()는 computational graph를 삭제 시켜줌
def metric_acc(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    return acc

def metric_mcc(y_pred, y_true):
    perc_y_pred = y_pred.detach().cpu().numpy()
    perc_y_true = y_true.detach().cpu().numpy()
    mcc = matthews_corrcoef(perc_y_true, perc_y_pred)
    return mcc