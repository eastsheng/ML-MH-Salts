from sklearn.svm import SVR
from sklearn.datasets import make_regression

# 创建模拟的回归数据集
X, y = make_regression(n_samples=100, n_features=1, random_state=0)

# 创建支持向量回归模型
svr = SVR(kernel='linear')

# 训练模型
svr.fit(X, y)

# 预测
y_pred = svr.predict(X)