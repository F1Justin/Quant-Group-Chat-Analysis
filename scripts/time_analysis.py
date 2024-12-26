import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import seaborn as sns
import os


# 读取origin_group_data目录下所有csv文件
directory = "/data_analysis/origin_group_data"
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

# 合并所有csv文件的数据
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# 过滤掉指定的用户
filtered_user_ids = [3583860171, 3639364238, 1424912867, 985393579]
df = df[~df['user_id'].isin(filtered_user_ids)]

# 转换时间字段为 datetime 格式 (UTC)
df['time'] = pd.to_datetime(df['time'])

# 转换为东八区时间
df['time'] = df['time'] + pd.Timedelta(hours=8)

# 按小时统计消息数量
hourly_counts = df.groupby(df['time'].dt.hour).size()
print("Hourly Message Distribution:")
print(hourly_counts)

# 绘制柱状图
plt.bar(hourly_counts.index, hourly_counts.values, color='skyblue')
plt.title('Hourly Message Distribution')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Messages')
plt.xticks(range(24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # 自动调整布局
plt.savefig('HourlyMessageDistribution.pdf')
plt.show()

# 提取工作日和周末信息
df['weekday'] = df['time'].dt.dayofweek  # 周一=0, 周日=6
weekday_counts = df.groupby('weekday').size()

# 打印工作日和周末的消息数量
print("Message Distribution by Weekday:")
print(weekday_counts)

# 绘制折线图
plt.plot(weekday_counts.index, weekday_counts.values, marker='o', color='orange')
plt.title('Message Distribution by Weekday')
plt.xlabel('Day of the Week (0=Monday)')
plt.ylabel('Number of Messages')
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()  # 自动调整布局
plt.savefig('MessageDistributionbyWeekday.pdf')
plt.show()

# 按天统计消息数量
daily_counts = df.groupby(df['time'].dt.date).size()

# 按周统计消息数量
weekly_counts = df.groupby(df['time'].dt.to_period('W')).size()

# 打印每周的消息数量
print("Weekly Message Volume:")
print(weekly_counts)

# 绘制时间序列图
weekly_labels = [f"{(period.start_time + pd.Timedelta(hours=8)).strftime('%m.%d')}-{(period.end_time + pd.Timedelta(hours=8)).strftime('%m.%d')}" for period in weekly_counts.index]
plt.plot(weekly_labels, weekly_counts.values, color='purple', marker='o', linestyle='--')
plt.title('Weekly Message Volume')
plt.xlabel('Week')
plt.ylabel('Number of Messages')
plt.grid(linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()  # 自动调整布局
plt.savefig('WeeklyMessageVolume.pdf')
plt.show()

# 提取星期和小时
df['weekday'] = df['time'].dt.weekday + 1  # 星期一为1，星期天为7
df['hour'] = df['time'].dt.hour

# 统计每小时在每一天的消息数量并计算平均值
heatmap_data = df.groupby(['hour', 'weekday']).size().unstack(fill_value=0)
# 获取每个星期几的天数
days_per_weekday = df.groupby('weekday').agg({'time': lambda x: len(x.dt.date.unique())})
# 对每列（星期几）除以对应的天数得到平均值
for weekday in range(1, 8):
    heatmap_data[weekday] = heatmap_data[weekday] / days_per_weekday.loc[weekday, 'time']

# 创建热力图
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='magma', annot=False, fmt='d', cbar=True)

# 图表美化
plt.title('Message Count Heatmap by Hour and Weekday')
plt.xlabel('Weekday (1=Monday, 7=Sunday)')
plt.ylabel('Hour of Day')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
plt.yticks(rotation=0)

# 显示图表
plt.tight_layout()
plt.show()

# 计算总消息数、总用户数、总群组数
total_messages = len(df)
total_users = df['user_id'].nunique()
total_groups = df['group_id'].nunique()

print(f"Total Messages: {total_messages}")
print(f"Total Users: {total_users}")
print(f"Total Groups: {total_groups}")

# 计算各群组的消息分布
group_message_counts = df.groupby('group_id').size()
print("Message Distribution by Group:")
print(group_message_counts)
# 绘制各群组的消息分布柱状图（对数坐标系）
plt.figure(figsize=(12, 6))
colors = plt.cm.YlGnBu(group_message_counts / max(group_message_counts))
group_message_counts.plot(kind='bar', color=colors)
plt.yscale('log')  # 设置y轴为对数坐标系
plt.title('Message Distribution by Group (Log Scale)')
plt.xlabel('Group ID')
plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees and align them
plt.ylabel('Number of Messages (Log Scale)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout(pad=1.5)  # Increase padding around the plot
plt.savefig('MessageDistributionbyGroup(LogScale).pdf')
plt.show()