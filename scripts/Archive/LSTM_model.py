import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 读取需要的列
data = pd.read_csv("scta_data_r.csv", usecols=[2, 3], names=['message_count', 'datetime'])

# 确保 'datetime' 列为日期时间格式
data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')  # 使用 'coerce' 来处理任何错误的格式
# 删除包含 NaT（非时间）的行，这些行可能是由于强制转换而产生的
data = data.dropna(subset=['datetime'])

# 检查日期时间索引是否正确设置
print(data.head())  # 检查前几行以确保正确加载

# 设置 datetime 为索引
data = data.set_index('datetime')

# 数据预处理
values = data['message_count'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# 将数据分成训练集和测试集（90/10 分割）
train_size = int(len(scaled_data) * 0.9)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# 为 LSTM 创建序列
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 24  # 序列长度
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# 将输入重塑为 [样本数，时间步长，特征数]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation='relu'))  # 使用 ReLU 激活函数
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=48, verbose=1) 

# 进行预测
y_pred = model.predict(X_test)

# 逆变换以获得原始尺度下的预测值
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print(f'均方误差：{mse:.2f}')
print(f'均方根误差：{rmse:.2f}')
print(f'平均绝对误差：{mae:.2f}')

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='true')
plt.plot(y_pred, label='foremost')
plt.xlabel('timestep')
plt.ylabel('msg_count')
plt.title('LSTM_scta')
plt.legend()
plt.show()
