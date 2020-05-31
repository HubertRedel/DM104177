import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, send_file, render_template
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

app = Flask(__name__)
K.set_image_data_format('channels_first')


def cnn_model():
    model = Sequential()
    model.add(Dense(13, activation='relu', input_shape=(15,)))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


def plot(history, result_folder):
    # Plot training & validation accuracy values
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Wykres skuteczności imiona polskie ')
    plt.ylabel('Skutecznoość')
    plt.xlabel('Epoka', fontsize=7)
    plt.legend(["Train (max = {0:.2f})".format(100 * max(history.history["acc"])),
                "Test (max = {0:.2f})".format(100 * max(history.history["val_acc"]))], loc='upper left')
    plt.grid(True)
    plt.xticks(range(0, len(history.history['acc']), 10))
    plt.savefig(os.path.join(result_folder, "model_acc.svg"), format="png", dpi=300, bbox_inches='tight')

    # Plot training & validation loss values
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss AlexNet')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.xticks(range(0, len(history.history['acc'])))
    plt.savefig(os.path.join(result_folder, "model_loss.svg"), format="svg", dpi=100, bbox_inches='tight')


@app.route("/")
def hello():
    return render_template("Home.html")


@app.route("/imiona_polskie.csv")
def get_csv():
    return send_file("imiona_polskie.csv")


@app.route("/start")
def learn():
    myData = pd.read_csv('imiona_polskie.csv', sep=';', header=None)

    X = np.array(myData[0])
    for a in range(0, X.size):
        X[a] = X[a].upper().rjust(15, '0')

    d = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for a in X.tolist():
        d = np.vstack((d, np.array([ord(c) for c in a])))

    X = np.delete(d, 0, 0)
    Y = to_categorical(myData[1])

    model = cnn_model()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    nb_epoch = 100
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=32)
    cb = TensorBoard()
    history = model.fit(X_train, Y_train,
                        epochs=nb_epoch,
                        validation_data=(X_val, Y_val),
                        callbacks=[cb]
                        )

    model.summary()
    model.count_params()
    plot(history, "")
    return send_file("model_acc.svg", mimetype='image/gif')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
