import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config):

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save('model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)




def train_seas(models, X_train, y_train, name, config):

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)





def main(target_model):

    config = {"batch": 10, "epochs": 25000}
    train = 'data/train/'
    test = 'data/test/'
    X_train, y_train, _, _ = process_data(train, test)

    if target_model == 'lstm':
        m = model.get_lstm([58, 64, 64, 2])
        train_model(m, X_train, y_train, target_model, config)
    if target_model == 'gru':
        m = model.get_gru([58, 64, 64, 2])
        train_model(m, X_train, y_train, target_model, config)
    if target_model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        m = model.get_saes([4800, 400, 400, 400, 2])
        train_seas(m, X_train, y_train, target_model, config)

if __name__ == '__main__':
    main('lstm')
