import math
import warnings
import pandas as pd
from data.data import process_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm


def MAPE(y_true, y_pred):
    arousal = abs(y_true[0] - y_pred[0]) / y_true[0]
    valence = abs(y_true[1] - y_pred[1]) / y_true[1]

    return arousal*100, valence*100


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    arousal, valence = MAPE(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    print('arousal error:%f%%' % arousal)
    print('valence error:%f%%' % valence)
    print('mse:%f' % mse)




def checkTestError():
    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']

    lag = 12
    train = 'data/train/'
    test = 'data/test/'
    _, _, X_test, y_test = process_data(train, test)

    y_preds = []
    for name, model in zip(names, models):
        print(name)
        if name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], -1))
        file = 'images/' + name + '.png'
        #plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        j = 1
        for i in range(len(predicted)):
            print('文件',j,': ')
            j += 1
            eva_regress(y_test[i], np.array(predicted[i]))




def printPlot(model):
    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    #saes = load_model('model/saes.h5')

    train = 'data/train/'
    test = 'data/test/'
    _, _, X_test, y_test = process_data(train, test)

    if model == 'saes':
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        #predicted = saes.predict(X_test)
    if model == 'lstm':
        predicted = lstm.predict(X_test)
    if model == 'gru':
        predicted = gru.predict(X_test)

    arousal_true, valence_true, arousal_pred, valence_pred = [], [], [], []
    for i in range(len(y_test)):
        arousal_true.append(y_test[i][0])
        valence_true.append(y_test[i][1])
        arousal_pred.append(round(predicted[i][0],2))
        valence_pred.append(round(predicted[i][1],2))


    plt.figure()
    plt.xlabel('Valence')
    plt.ylabel('Arousal')  # 设置坐标轴的文字标签

    ax = plt.gca()  # get current axis 获得坐标轴对象

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴

    ax.spines['bottom'].set_position(('data', 0))  # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['left'].set_position(('data', 0))

    plt.title('LALALA', fontsize=22, color=(0.4,0.4,0.4), loc='center')

    plt.plot(np.array(valence_true), np.array(arousal_true), linestyle='--', label='origin')
    plt.plot(np.array(valence_pred), np.array(arousal_pred),label='pred')
    plt.xlim(xmin=-6,xmax=6)
    plt.ylim(ymin=-6, ymax=6)

    plt.legend(bbox_to_anchor=(1.05, 0), loc=0, borderaxespad=0)

    plt.savefig('images/result.png')

if __name__ == '__main__':
    #checkTestError()
    printPlot('gru')
