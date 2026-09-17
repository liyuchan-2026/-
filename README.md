import numpy as np

#制造训练数据
np.random.seed(1)
n = 120
x = np.random.randn(n)
y = 2 * x ** 2 + 2 * x + 1 + 0.01 * np.random.randn(n)  #原函数为y=2x^+2x+1

#初始化参数
w1 = 0.0
w2 = 0.0
b = 0.0
lr = 0.05
epochs = 100

#设置梯度下降循环
for i in range(epochs):
    y_pred = w1 * x ** 2 + w2 * x + b
    dw1 = 0.0
    dw2 = 0.0
    db = 0.0
    for z in range(n):
        error = y_pred[z] - y[z]
        dw1 += error * (x[z] ** 2)
        dw2 += error * x[z]
        db += error
    dw1 = dw1 / n
    dw2 = dw2 / n
    db = db / n

    #更新参数（斜率）
    w2 = w2 - lr * dw2
    b = b - lr * db
    w1 = w1 - lr * dw1

    #计算损失（引入损失方程）
    loss = 0.0
    for z in range(epochs):
        loss = loss + (y_pred[z] - y[z]) ** 2
    loss = loss / (2 * n)
    if (i + 1) % 5 == 0:
        print(f"轮次{i + 1}, loss={loss:.6f}")

#得出结果
print("训练得到w1=", round(w1, 4))
print("训练得到w2=", round(w2, 4))
print("训练得到b=", round(b, 4))
print("真实值 w1=2, w2=2, b=1")
