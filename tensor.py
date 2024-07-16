import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


# 같은 디렉토리에 있는 파일 경로 설정
path = "data-02-stock_daily.csv"

# 파일 경로가 존재하는지 확인
if os.path.exists(path):
    data = pd.read_csv(path, header=1)
    print(data.head())
else:
    print("파일이 존재하지 않습니다:", path)

# 데이터 시각화
fig = plt.figure(figsize=(30,20))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

ax1.plot(data['Open'], label="Open")
ax1.plot(data['High'], label="High")
ax1.plot(data['Low'], label="Low")
ax1.plot(data['Close'], label="Close")

ax1.legend()

ax2.plot(data['Volume'], label="Volume")
ax2.legend()

ax3.plot(data['Open'][0:7], linewidth=3.0, label="Open")
ax3.plot(data['High'][0:7], linewidth=3.0, label="High")
ax3.plot(data['Low'][0:7], linewidth=3.0, label="Low")
ax3.plot(data['Close'][0:7], linewidth=3.0, label="Close")
ax3.legend(prop={'size': 30})

# Open, High, Low, Volume으로 Close 가격 예측하기 
xdata = data[["Open", "High", "Low", "Volume"]]
ydata = pd.DataFrame(data["Close"])

# 데이터 표준화
xdata_ss = StandardScaler().fit_transform(xdata).astype(np.float32)
ydata_ss = StandardScaler().fit_transform(ydata).astype(np.float32)

print(xdata_ss.shape, ydata_ss.shape)
# >>> (732, 4) (732, 1)

# 트레이닝 테스트 데이터 분리 
xtrain = xdata_ss[220:,:]
xtest = xdata_ss[:220,:]
ytrain = ydata_ss[220:,:]
ytest = ydata_ss[:220,:]

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
# >>> (512, 4) (512, 1) (220, 4) (220, 1)

# 변수 선언 
w = tf.Variable(tf.random.normal([4, 1], dtype=tf.float32)) # 1: 출력되는 y의 갯수 
b = tf.Variable(tf.random.normal([1], dtype=tf.float32)) # 1: 출력되는 y의 갯수 

# 가설 함수 
def model(x):
    return tf.matmul(x, w) + b

# 손실 함수 
def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 옵티마이저
optimizer = tf.optimizers.Adam(learning_rate=0.01)

# 트레이닝 함수
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_function(y, predictions)
    gradients = tape.gradient(loss, [w, b])
    optimizer.apply_gradients(zip(gradients, [w, b]))
    return loss

# 트레이닝
for step in range(2001):
    loss = train_step(xtrain, ytrain)
    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss.numpy()}")

# 테스트
predictions = model(xtest)
evaluate = np.mean(np.square(predictions.numpy() - ytest))
print(f"Evaluate: {evaluate}")

# 예측 결과 시각화
tf_predicted = np.dot(xdata_ss, w.numpy()) + b.numpy()

plt.figure()
plt.plot(tf_predicted[:220], label="predict")
plt.plot(ytest, label="actual")

plt.legend(prop={'size': 20})
plt.show()
