import pandas as pd
import numpy as np

# 读取CSV文件，替换为你实际的文件路径
df = pd.read_csv('origin_scta.csv')

# 将time字段转换为日期时间类型，根据实际时间格式调整参数
df['time'] = pd.to_datetime(df['time'])

# 转换为东八区时间
df['time'] = df['time'] + pd.Timedelta(hours=8)

# 按照实际数据统计消息数量
grouped = df.copy()
grouped['date'] = grouped['time'].dt.date
grouped['hour'] = grouped['time'].dt.hour
grouped = grouped.groupby(['date', 'hour']).size().reset_index(name='message_count')
grouped['message_count'] = grouped['message_count'].astype(int)

# Convert 'date' column to the same type as in full_df
grouped['date'] = pd.to_datetime(grouped['date'])

# Generate full date-hour index
min_date = grouped['date'].min()
max_date = grouped['date'].max()
date_range = pd.date_range(start=min_date, end=max_date)
hour_range = range(24)  # Simpler range creation

# Efficiently create the full DataFrame using reindex
full_df = pd.MultiIndex.from_product([date_range, hour_range], names=['date', 'hour']).to_frame(index=False)

# Merge with the grouped data
full_df = full_df.merge(grouped, on=['date', 'hour'], how='left').fillna(0)

# Convert 'message_count' to int
full_df['message_count'] = full_df['message_count'].astype(int)

# 调整列顺序，使其更符合习惯（可选步骤）
full_df = full_df[['date', 'hour', 'message_count']]

# 将结果保存为新的CSV文件，可根据需求修改文件名和路径
full_df.to_csv('scta_data.csv', index=False)
