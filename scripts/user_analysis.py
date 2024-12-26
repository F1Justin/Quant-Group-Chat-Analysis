import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import numpy as np
import os

# 读取origin_group_data目录下所有csv文件
directory = "/data_analysis/origin_group_data"
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

# 合并所有csv文件的数据
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# 过滤掉指定的用户
filtered_user_ids = [1234, 5678, 91011]
df = df[~df['user_id'].isin(filtered_user_ids)]

# 转换时间字段为 datetime 格式 (UTC)
df['time'] = pd.to_datetime(df['time'])

# 转换为东八区时间
df['time'] = df['time'] + pd.Timedelta(hours=8)

# 提取小时字段
df['hour'] = df['time'].dt.hour

# ---- 1. 按用户统计消息数量 ----
user_activity = df.groupby('user_id').size().reset_index(name='message_count')
user_activity = user_activity.sort_values(by='message_count', ascending=False)

# 消息数量分布直方图
plt.figure(figsize=(10, 6))
plt.hist(user_activity['message_count'], bins=200, color='skyblue', edgecolor='black')
plt.xscale('linear')
plt.yscale('log')
plt.title('User Message Count Distribution')
plt.xlabel('Message Count')
plt.ylabel('Number of Users (Log Scale)')
# 打印消息数量的基本统计信息
print("\n消息数量统计:")
print(f"用户总数: {len(user_activity)}")
print(f"平均消息数: {user_activity['message_count'].mean():.2f}")
print(f"中位数消息数: {user_activity['message_count'].median():.2f}")
print(f"最大消息数: {user_activity['message_count'].max()}")
print(f"最小消息数: {user_activity['message_count'].min()}")
# 计算标准差和分位数
print(f"标准差: {user_activity['message_count'].std():.2f}")
print(f"25%分位数: {user_activity['message_count'].quantile(0.25):.2f}")
print(f"75%分位数: {user_activity['message_count'].quantile(0.75):.2f}")


# 计算发送超过100条消息的用户比例
active_users = len(user_activity[user_activity['message_count'] > 100])
total_users = len(user_activity)
print(f"\n发送超过100条消息的用户比例: {(active_users/total_users)*100:.2f}%")
# 创建消息数量区间
bins = range(0, int(user_activity['message_count'].max()) + 50, 50)
message_ranges = pd.cut(user_activity['message_count'], bins=bins)
range_counts = message_ranges.value_counts().sort_index()
# 过滤掉计数为0的区间
range_counts = range_counts[range_counts > 0]

print("\n每50条消息区间的用户数量:")
for interval, count in range_counts.items():
    print(f"{interval}: {count}人")
plt.savefig('NumberofUsers(LogScale).pdf')
plt.show()


# ---- 2. 用户活跃时段分布 (小时热力图  - 仅前25名用户 ----
top_users = user_activity.head(25)
top_user_ids = top_users['user_id'].values

df_top_users = df[df['user_id'].isin(top_user_ids)]
hourly_counts = df_top_users.groupby(['user_id', 'hour']).size().unstack(fill_value=0)

hourly_counts = hourly_counts.loc[top_user_ids]  # 保持用户顺序
hourly_counts = hourly_counts.T  # 转置以方便绘图

# 定义用户的调色板
cmap_top10 = get_cmap('Set3')      # 前 10 名用户渐变
cmap_1020 = get_cmap('Pastel1')   # 接下来 10 名用户渐变
cmap_others = get_cmap('Pastel2') # 剩下 5 名用户渐变

# 分别生成颜色列表
num_top10 = 10
num_1020 = 10
num_others = 5

colors_top10 = [cmap_top10(i / num_top10) for i in range(num_top10)]
colors_1020 = [cmap_1020(i / num_1020) for i in range(num_1020)]
colors_others = [cmap_others(i / num_others) for i in range(num_others)]

# 合并颜色列表
custom_colors = colors_top10 + colors_1020 + colors_others

# 绘图
plt.figure(figsize=(15, 8))
hourly_counts.plot(kind='bar', stacked=True, figsize=(15, 8), color=custom_colors)
plt.title('Hourly Activity Distribution (Top 25 Users)')
plt.xlabel('Hour of Day')
plt.ylabel('Message Count')
plt.legend(title='User ID', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)  # 移动图例到外面
plt.xticks(rotation=0)
plt.tight_layout()  # 自动调整布局
plt.show()

# ---- 3. 用户在一周内的活跃情况 (星期热力图)  - 仅前25名用户 ----
df['weekday'] = df['time'].dt.weekday + 1  # 1=星期一, 7=星期日
weekday_activity = df.groupby(['user_id', 'weekday']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
sns.heatmap(hourly_counts, cmap='YlGnBu', cbar=True)
plt.title('Hourly Activity Heatmap (Top 25 Users)')
plt.xlabel('User ID (Descending Order of Activity)')
plt.ylabel('Hour of Day')
plt.tight_layout()  # 自动调整布局
plt.show()

# ---- 4. 分类用户活跃度 ----
# 平均消息数量
average_message_count = user_activity['message_count'].mean()

# 定义分类函数
def classify_activity(count):
    if count < 0.1 * average_message_count:
        return '不活跃'
    elif 0.1 * average_message_count <= count < average_message_count:
        return '低活跃'
    elif average_message_count <= count < 10 * average_message_count:
        return '活跃'
    else:
        return '水群领域大神'

# 应用分类函数
user_activity['activity_level'] = user_activity['message_count'].apply(classify_activity)

# 活跃用户数量统计
activity_levels = user_activity['activity_level'].value_counts()

# 各活跃度等级总消息数量
message_by_level = user_activity.groupby('activity_level')['message_count'].sum()

# 定义每个活跃度级别对应的消息范围
ranges = {
    '不活跃': f'1 - {int(0.1 * average_message_count)}',
    '低活跃': f'{int(0.1 * average_message_count)} - {int(average_message_count)}',
    '活跃': f'{int(average_message_count)} - {int(10 * average_message_count)}',
    '水群领域大神': f'>{int(10 * average_message_count)}'
}

# 生成柱状图
plt.figure(figsize=(8, 5))
plt.rcParams['font.sans-serif'] = ['Noto Sans SC']  # 设置字体以显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 绘制柱状图
bar_colors = ['#68C1EE', '#9881F3', '#9D3F9D', '#C22762']
activity_levels.plot(kind='bar', color=bar_colors, edgecolor='black')
plt.title('用户活跃度分布')
plt.xlabel('活跃度级别')
plt.ylabel('用户数量')

# 添加消息数量范围和总消息数到 x 轴标签
labels = [
    f'{label}\n({ranges[label]} 条)\n总计贡献了 {int(message_by_level[label]):,} 条'
    for label in activity_levels.index
]
plt.xticks(range(len(labels)), labels, rotation=0)

# 在每个柱子顶上标出用户数量
for i, count in enumerate(activity_levels.values):
    plt.text(i, count + 1, f'{count}', ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()  # 自动调整布局
plt.savefig('activity_levels.pdf')
plt.show()

# 计算累积比例
user_activity['cumulative_ratio'] = user_activity['message_count'].cumsum() / user_activity['message_count'].sum()

# 选择前 20 名用户及“其他”
top_users = user_activity.head(20)
other_ratio = 1 - top_users['cumulative_ratio'].iloc[-1]

# 合并为“其他用户”
labels = list(top_users['user_id']) + ['其他']
sizes = list(top_users['message_count']) + [user_activity['message_count'].sum() - top_users['message_count'].sum()]
# 假设 sizes 和 labels 已经定义
sizes = np.array(sizes)  # 确保 sizes 是数组类型
x_positions = np.arange(len(labels))  # 生成索引位置

# 绘制瀑布图
cumulative = np.cumsum(sizes)
cumulative = np.insert(cumulative, 0, 0)[:-1]  # 计算前置累积值

plt.figure(figsize=(12, 6))
plt.bar(x_positions, sizes, width=0.6, color=plt.cm.tab20.colors[:len(labels)])
plt.plot(x_positions, cumulative, marker='o', color='red', label='累积比例')
plt.title('用户消息贡献瀑布图')
plt.xlabel('用户')
plt.ylabel('消息数')
plt.xticks(x_positions, labels, rotation=45)  # 设置标签
plt.legend()
plt.show()