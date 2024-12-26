import os
import pandas as pd
import matplotlib.pyplot as plt
from multiprocess import Pool, cpu_count
from matplotlib import rcParams

# 设置字体和全局配置
rcParams['font.sans-serif'] = ['Noto Sans SC']
rcParams['axes.unicode_minus'] = False

# 蓝色系配色
colors = ['#1f77b4', '#aec7e8', '#6baed6', '#2171b5']

# 定义读取和处理文件的函数
def process_file(file):
    df = pd.read_csv(file)
    # 解析并分类消息类型
    message_counts = {
        "文本消息": 0,
        "图片消息": 0,
        "表情消息": 0,
        "卡片消息（json）": 0
    }

    filtered_user_ids = [12345, 67890]
    df = df[~df['user_id'].isin(filtered_user_ids)]
    
    for _, row in df.iterrows():
        try:
            message_data = eval(row['message'])
            # 判断类型
            for message_part in message_data:
                if message_part['type'] == 'image':
                    if 'summary' in message_part['data'] and message_part['data']['summary'] == '[动画表情]':
                        message_counts["表情消息"] += 1
                    else:
                        message_counts["图片消息"] += 1
                elif message_part['type'] == 'text':
                    message_counts["文本消息"] += 1
                else:
                    message_counts["卡片消息（json）"] += 1
        except Exception:
            message_counts["卡片消息（json）"] += 1
    return message_counts

# 目录路径
directory = "/Users/justin/Desktop/data_analysis/origin_group_data"
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

# 多核处理
with Pool(cpu_count()) as pool:
    results = pool.map(process_file, all_files)

# 汇总结果
total_counts = {"文本消息": 0, "图片消息": 0, "表情消息": 0, "卡片消息（json）": 0}
for result in results:
    for key in total_counts:
        total_counts[key] += result[key]

# 绘制饼图
labels = list(total_counts.keys())
sizes = list(total_counts.values())
explode = (0.1, 0.1, 0, 0)  # 突出文本消息和图片消息
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    sizes,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops=dict(width=0.3, edgecolor='w'),  # 环形图，增加白色边框
    pctdistance=0.8,
    textprops=dict(color='black')
)

plt.title("消息类型分布", fontsize=16, fontweight='bold', color='#333333', pad=20)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')


plt.axis('equal')  # 保持饼图为圆形
plt.tight_layout()
plt.savefig('MessageTypeDistribution.pdf', bbox_inches='tight')
plt.show()