import numpy as np
import matplotlib.pyplot as plt

# 定义 LogisticRegression 类
class LogisticRegression:

    def __init__(self):

        # 定义学习率
        self.learning_rate: float = 0

        # 定义迭代次数，即模型进行梯度下降的轮数
        self.num_iterations: int = 0

        # self.weights 代表模型中各个特征的权重
        self.weights: list[float] = []

        # self.bias 是模型中的偏置项
        self.bias: float = 0

    @staticmethod
    def sigmoid(z):
        """
        定义 sigmod 函数
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y, learning_rate: float = 0.1, num_iterations: int = 5000):
        """
        定义训练模型
        :param x:
        :param y:
        :return:
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        # 样本数量、特征向量（x 是一个二维数组，代表训练数据的特征矩阵。每一行对应一个样本，每一列对应一个特征）
        num_samples, num_features = x.shape

        # self.weights 初始化为零向量
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):

            # 计算线性模型的输出
            prediction_vector = self.predict_vector(x)

            # 通过 sigmoid 函数将线性模型的输出转换为概率值
            y_pred = LogisticRegression.sigmoid(prediction_vector)

            # 计算权重、偏置的梯度
            dw = (1 / num_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # 使用梯度下降更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x):
        """
        定义预测函数
        """

        # 计算线性模型的输出
        prediction_vector = self.predict_vector(x)

        # 通过 sigmoid 函数转换为概率值
        y_pred = LogisticRegression.sigmoid(prediction_vector)

        # 根据概率值进行分类，概率大于 0.5 的样本预测为 1，否则预测为 0
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_cls)

    def predict_vector(self, x):
        return np.dot(x, self.weights) + self.bias


def main():
    test_data_proportion: float = 0.2

    # Generate samples.

    # 设置随机数
    np.random.seed(42)

    # 两个特征的均值和方差
    mean_1 = [2, 2]
    cov_1 = [[2, 0], [0, 2]]
    mean_2 = [-2, -2]
    cov_2 = [[1, 0], [0, 1]]

    # 生成类别 1 的样本
    x1 = np.random.multivariate_normal(mean_1, cov_1, 100)
    y1 = np.zeros(100)

    # 生成类别 2 的样本
    x2 = np.random.multivariate_normal(mean_2, cov_2, 100)
    y2 = np.ones(100)

    # 合并样本和标签
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2))

    # 随机打乱样本和标签的顺序
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # 手动划分训练集和测试集，这里设置测试集比例为 0.2
    test_samples = int(len(x) * test_data_proportion)
    x_train = x[:-test_samples]
    y_train = y[:-test_samples]
    x_test = x[-test_samples:]
    y_test = y[-test_samples:]

    # Train the model.

    # 创建逻辑回归对象并训练模型
    model = LogisticRegression()
    model.fit(x_train, y_train, learning_rate=0.1, num_iterations=5000)


    # Test the model.

    # 在测试集上进行预测
    test_predictions = model.predict(x_test)
    print(test_predictions)

    # 计算准确率
    accuracy = np.mean(test_predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("Trained weights:", model.weights)
    print("Trained bias:", model.bias)

    # Draw the scatter.
    # 绘制决策边界
    x_values = np.array([x[:, 0].min() - 1, x[:, 0].max() + 1])
    y_values = -(model.weights[0] * x_values + model.bias) / model.weights[1]
    plt.plot(x_values, y_values, label='Decision Boundary', color='blue')

    # 绘制散点图
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression with Decision Boundary')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()