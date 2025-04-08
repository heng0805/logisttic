import numpy as np

class LogisticRegression:
    #定义 LogisticRegression 类
    def __init__(self, learning_rate=0.1, num_iterations=5000):
        self.learning_rate = learning_rate #定义学习率
        self.num_iterations = num_iterations #定义迭代次数，即模型进行梯度下降的轮数
        self.weights = None #self.weights 代表模型中各个特征的权重
        self.bias = None #self.bias 是模型中的偏置项

    def sigmoid(self, z):
        #定义sigmod函数
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        #定义训练模型
        num_samples, num_features = X.shape #样本数量、特征向量（X 是一个二维数组，代表训练数据的特征矩阵。每一行对应一个样本，每一列对应一个特征）
        self.weights = np.zeros(num_features) #self.weights 初始化为零向量
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias #计算线性模型的输出
            y_pred = self.sigmoid(linear_model) #通过 sigmoid 函数将线性模型的输出转换为概率值

            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y)) #计算权重、偏置的梯度
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw #使用梯度下降更新权重和偏置
            self.bias -= self.learning_rate * db

    def predict(self, X): #定义预测函数
        linear_model = np.dot(X, self.weights) + self.bias #计算线性模型的输出
        y_pred = self.sigmoid(linear_model) #通过 sigmoid 函数转换为概率值
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred] #根据概率值进行分类，概率大于 0.5 的样本预测为 1，否则预测为 0
        return np.array(y_pred_cls)


def main():
    # 设置随机数
    np.random.seed(42)

    # 两个特征的均值和方差
    mean_1 = [2, 2]
    cov_1 = [[2, 0], [0, 2]]
    mean_2 = [-2, -2]
    cov_2 = [[1, 0], [0, 1]]

    # 生成类别1的样本
    X1 = np.random.multivariate_normal(mean_1, cov_1, 10)
    y1 = np.zeros(10)

    # 生成类别2的样本
    X2 = np.random.multivariate_normal(mean_2, cov_2, 10)
    y2 = np.ones(10)

    # 合并样本和标签
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2))

    # 随机打乱样本和标签的顺序
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # 创建逻辑回归对象并训练模型
    model = LogisticRegression(learning_rate=0.1, num_iterations=5000)
    model.fit(X, y)

    # 进行预测
    predictions = model.predict(X)
    print(predictions)


if __name__ == "__main__":
    main()
