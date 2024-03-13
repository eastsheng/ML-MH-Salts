import pandas as pd
import numpy as np
from src import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import fastdataing as fd
from sklearn.model_selection import cross_val_score


def data_process(df):
    labelencoder_B = LabelEncoder()
    df["Inhibitors"] = labelencoder_B.fit_transform(df["Inhibitors"])
    x = df[['Inhibitors', 'Conc(wt%)', 'Pressure (MPa)', 'Temp(K)',
       'C1(%)', 'C2(%)', 'C3(%)', 'nC4(%)', 'iC4(%)', 'H2S(%)', 'MW', 'LogP',
       'NumHDonors', 'NumHAcceptors', 'N2(%)', 'CO2(%)']].values
    y = df[['Induction delay time (min)']].values

    return x,y

def metrics_score(estimator,x_test,y_test,y_predict):
    # score = estimator.score(x_test, y_test)
    score = cross_val_score(estimator, x_test, y_test, cv=5, scoring='neg_mean_squared_error')
    RS = metrics.r2_score(y_test,y_predict)
    MSE = metrics.mean_squared_error(y_test, y_predict)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_predict))
    print("每折交叉验证的 MSE scores: ", score)
    print("R^2:", RS)
    print("MSE:", MSE)
    print("RMSE:", RMSE)
    return RS, MSE, RMSE

def main():
    df = pd.read_csv("./data/InduceTime_PVP.csv")

    x,y = data_process(df)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=22)
    transfer = MinMaxScaler() # # Min-Max标准化
    # transfer = StandardScaler() # # StandardScaler
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    estimator = SGDRegressor(max_iter=100)

    estimator.fit(x_train,y_train.ravel())
    y_predict = estimator.predict(x_test)
    RS, MSE, RMSE = metrics_score(estimator,x_test,y_test.ravel(),y_predict.ravel())
    fig = fd.add_fig(figsize=(10,8))
    ax = fd.add_ax(fig,subplot=(111))
    ax.scatter(y_test,y_predict,color="b")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Exp Induce Time (min)")
    ax.set_ylabel("Predict Induce Time (min)")
    ax.text(0,175,s=r"$\regular R^2$ = "+str(round(RS,4)))
    ax.text(0,160,s=r"$\regular MSE$ = "+str(round(MSE,4)))
    ax.text(0,145,s=r"$\regular RMSE$ = "+str(round(RMSE,4)))
    ax.set_title("SGDRegressor")
    plt.savefig("./imgs/SGDRegressor_PVP.png",dpi=300,transparent=True)
    plt.show()
    return



if __name__ == "__main__":
    main()