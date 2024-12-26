import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

def time_of_day(ts):
    '''
    将时间戳转化为时间段
    '''
    hour = pd.to_datetime(ts).hour
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"

def process_user_features(user_ids, df, vectorizer):
    """
    进程池执行的函数，提取每个用户的特征
    """
    user_text_feature = {}
    # 获取这些用户的子集数据
    df_subset = df[df['user_id'].isin(user_ids)]
    text_features = vectorizer.transform(df_subset['plain_text'].fillna(''))
    
    # 初始化每个用户的特征为零向量
    for user_id in user_ids:
        user_text_feature[user_id] = np.zeros(text_features.shape[1])
    
    # 按 user_id 分组并累加特征
    for i, (_, row) in enumerate(df_subset.iterrows()):
        user_text_feature[row['user_id']] += text_features[i].toarray()[0]
    
    return user_text_feature

def extract_features(df, num_processes=cpu_count()):
    '''
    提取用户行为特征
    '''
    df['time'] = pd.to_datetime(df['time'])  # 将时间转换为时间类型
    df['time_of_day'] = df['time'].apply(time_of_day)
    
    user_daily_counts = df.groupby([df['time'].dt.date, 'user_id']).size().unstack(fill_value=0)
    user_total_counts = user_daily_counts.sum(axis=0)
    user_time_of_day_counts = df.groupby(['user_id', 'time_of_day']).size().unstack(fill_value=0)
    
    # 对文本特征进行提取，这里提取 top 10 关键词
    vectorizer = TfidfVectorizer(stop_words=None, max_features=10)
    text_features = vectorizer.fit_transform(df['plain_text'].fillna(''))

    feature_df = pd.DataFrame(user_total_counts.rename('total_counts'))
    feature_df = feature_df.merge(user_time_of_day_counts, left_index=True, right_index=True, how='left').fillna(0)
    
    # 初始化文本特征为 0 向量
    feature_df["text_feature"] = [np.zeros(text_features.shape[1]).tolist() for _ in range(len(feature_df))]

    # 使用多进程计算文本特征
    user_ids = np.array_split(feature_df.index, num_processes)
    with Pool(num_processes) as pool:
         user_text_feature_list = pool.map(partial(process_user_features, df=df, vectorizer=vectorizer), user_ids)
    
    user_text_feature = {}
    for d in user_text_feature_list:
        user_text_feature.update(d)
    
    # 将 text_feature 中的数据进行聚合
    for k, v in user_text_feature.items():
         feature_df.at[k, "text_feature"] = v.tolist()
    
    # **确保只处理有效用户**
    valid_user_ids = df['user_id'].unique()
    feature_df = feature_df[feature_df.index.isin(valid_user_ids)]

    # 返回特征 DataFrame 和 vectorizer
    return feature_df, vectorizer

def kmeans_clustering(features_df, n_clusters=5):
    '''
    k-means 聚类
    '''
    # 标准化数值特征
    numerical_features = features_df.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(features_df[numerical_features])

    # 获取文本特征并合并
    text_features = np.array(features_df["text_feature"].tolist())
    combined_features = np.hstack((scaled_numerical_features, text_features))

    # 训练模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_df['cluster'] = kmeans.fit_predict(combined_features)

    return features_df

if __name__ == '__main__':
    # 读取目录下所有 CSV 文件
    directory = "/Users/justin/Desktop/data_analysis/origin_group_data"
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    
    try:
        start_time = time.time()
        df_list = [pd.read_csv(file) for file in all_files]
        df = pd.concat(df_list, ignore_index=True)
        
        # 处理 non-finite 值并将 user_id 列的数据类型转换为 int
        df['user_id'].fillna(-1, inplace=True)
        df['user_id'] = df['user_id'].astype(int)

        # 排除机器人用户和无效的 user_id (-1)
        bot_ids = [3583860171, 3639364238, 1424912867, 985393579, -1]
        df = df[~df['user_id'].isin(bot_ids)]
        
        print(f"读取并过滤数据耗时: {time.time() - start_time:.2f} 秒")
        print(f"数据形状: {df.shape}")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        exit()
    
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # 特征提取
    start_time = time.time()
    features_df, vectorizer = extract_features(df)
    print(f"特征提取耗时: {time.time() - start_time:.2f} 秒")
    
    # K-means 聚类
    start_time = time.time()
    clustered_df = kmeans_clustering(features_df, n_clusters=3)
    print(f"聚类耗时: {time.time() - start_time:.2f} 秒")
    
    # 保存结果到文件
    output_file_name = "clustered_output.csv"
    try:
        clustered_df.to_csv(output_file_name, index=True)
        print(f"聚类结果已保存到 {output_file_name}")
    except Exception as e:
        print(f"保存文件时发生错误：{e}")
        exit()
    
    # 打印每个簇的统计数据
    numerical_features = features_df.select_dtypes(include=np.number).columns.tolist()
    for cluster_id in sorted(clustered_df['cluster'].unique()):
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data)} users):")
        
        for col in numerical_features:
            print(f"  Avg {col}: {cluster_data[col].mean():.2f}")

            
def visualize_clusters(features_df, n_clusters=5, save_path="cluster_visualization.png"):
    """
    可视化聚类结果，并将图片保存到文件
    """
    # 确保要绘制的列存在于数据集中
    if 'total_counts' not in features_df.columns or 'morning' not in features_df.columns:
        print("Error: Columns 'total_counts' or 'morning' not found in the dataset. Please check your data.")
        return
    
    # 绘制散点图
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='total_counts',
        y='morning',  # 替换为实际的特征列
        hue='cluster',
        data=features_df,
        palette='viridis'
    )
    plt.title(f'K-Means Clustering (n_clusters={n_clusters})')
    plt.xlabel('Total Counts')
    plt.ylabel('Morning Activities')  # 替换为实际意义
    plt.legend(title='Cluster', loc='best')

    # 显示图形
    plt.show()

    # 保存图形
    try:
        plt.savefig(save_path)
        print(f"Cluster visualization saved to {save_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")

# 主程序调用 visualize_clusters
if __name__ == '__main__':
    visualize_clusters(clustered_df, n_clusters=3, save_path="cluster_visualization.png")