import pandas as pd

# 读取CSV文件，可根据实际情况调整encoding参数
data = pd.read_csv('scta_data.csv', encoding='utf-8')  

# 将date列转换为日期时间类型
data['date'] = pd.to_datetime(data['date'])

# 将hour列转换为字符串类型，并在格式上补全为两位数字（例如 1 变为 01）
data['hour'] = data['hour'].astype(str).str.zfill(2)

# 合并date和hour列，形成新的列，格式为 年/月/日/时
data['datetime'] = data['date'].dt.strftime('%Y/%m/%d') + '/' + data['hour']

# 查看合并后的结果
print(data['datetime'])

# 将处理后的数据保存为新的CSV文件
data.to_csv('scta_data_r.csv', index=False)