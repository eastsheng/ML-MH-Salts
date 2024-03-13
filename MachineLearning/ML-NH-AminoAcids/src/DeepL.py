import pandas as pd
import numpy as np
from src import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
import fastdataing as fd
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics,optimizers

print(tf.__version__)

def data_process(df):
    labelencoder_B = LabelEncoder()
    df["Inhibitors"] = labelencoder_B.fit_transform(df["Inhibitors"])

    X = df[['Conc(wt%)', 'Pressure (MPa)', 'Temp(K)',
       'C1(%)', 'C2(%)', 'C3(%)', 'nC4(%)', 'iC4(%)'
       ]]
    y = df['Induction delay time (min)']

    return X, y

def metrics_score(y_test,y_predict):
    RS = r2_score(y_test,y_predict)
    MSE = mean_squared_error(y_test, y_predict)
    RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
    print("R^2:", RS)
    print("MSE:", MSE)
    print("RMSE:", RMSE)
    return RS, MSE, RMSE


def plot_metric(fig,history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    ax = fd.add_ax(fig,subplot=(211))
    ax.plot(epochs, train_metrics, 'bo--',label="train_"+metric)
    ax.plot(epochs, val_metrics, 'ro-',label='val_'+metric)
    ax.set_title('Training and validation '+ metric)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric)

    return

def plot_predict(fig,y_test,y_predict):
    RS, MSE, RMSE = metrics_score(y_test,y_predict)
    ax = fd.add_ax(fig,subplot=(212))
    ax.scatter(y_test,y_predict,color="b")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',linewidth = 2.0)
    ax.text(0,y_test.max()-y_test.max()/15,s=r"$\regular R^2$ = "+str(round(RS,4)))
    ax.text(0,y_test.max()-2*(y_test.max()/15),s=r"$\regular MSE$ = "+str(round(MSE,4)))
    ax.text(0,y_test.max()-3*(y_test.max()/15),s=r"$\regular RMSE$ = "+str(round(RMSE,4)))
    ax.set_xlabel("Exp Induce Time (min)")
    ax.set_ylabel("Predict Induce Time (min)")


def main():
    # df = pd.read_csv("./data/InduceTime_all.csv")
    df = pd.read_csv("./data/InduceTime_PVP.csv")
    X,y = data_process(df)
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=0)
    # transfer = MinMaxScaler() # # Min-Max标准化
    transfer = StandardScaler() # # StandardScaler
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    
    tf.keras.backend.clear_session()
    initializer = tf.initializers.RandomNormal(stddev=0.01)
    # optimizer = optimizers.SGD(learning_rate=0.03)
    model = models.Sequential()
    # model.add(layers.Flatten())
    model.add(layers.Dense(10,activation = 'ReLU',input_shape=(8,),kernel_initializer=initializer))
    model.add(layers.Dense(10,activation = 'ReLU'))
    model.add(layers.Dense(1))
    # print(model.summary())
    model.compile(optimizer="Adam",loss="mse",metrics=["mae"])
    history = model.fit(
                        x_train,
                        y_train,
                        batch_size = 4,
                        epochs = 1000,
                        validation_split = 1/3,
                        # verbose=1,
                        shuffle=False
        )

    model.save('./data/save_models/pvp.h5')  
    # model = models.load_model('./data/save_models/pvp.h5')
    # model.evaluate(x_test,y_test)

    loss, acc = model.evaluate(x = x_test,y = y_test)
    tf.print("test data, accuracy:{:5.2f}%".format(100 * acc+0.1))
    
    y_predict = model.predict(x_test)

    fig = fd.add_fig(figsize=(10,16))

    plot_metric(fig,history,"loss")
    plot_predict(fig,y_test,y_predict)

    # plt.savefig("./imgs/DeepL_all.png",dpi=300,transparent=True)
    plt.savefig("./imgs/DeepL_PVP.png",dpi=300,transparent=True)
    plt.show()

    return



if __name__ == "__main__":
    main()
