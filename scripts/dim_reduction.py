import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import argparse  # 导入 argparse

def dimensionality_reduction(data_path, output_path, method='pca', n_components=2, image_output_path=None):
    """
    读取数据，进行降维，并输出降维后的数据和可视化图片。

    Args:
        data_path (str): 输入数据的 CSV 文件路径。
        output_path (str): 输出降维后数据的 CSV 文件路径。
        method (str, optional): 降维方法，'pca' 或 'tsne'，默认为 'pca'。
        n_components (int, optional): 降维后的维度数，默认为 2。
        image_output_path (str, optional): 可视化图片输出路径，默认为 None，如果不指定则不输出图片。
    """

    # 读取数据
    df = pd.read_csv(data_path)
    
    # 提取文本特征
    text_features = df['text_feature'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()
    text_features = np.array(text_features)
    
    # 进行降维
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=114514)
    else:
        raise ValueError("Invalid dimensionality reduction method. Choose 'pca' or 'tsne'.")
    
    reduced_features = reducer.fit_transform(text_features)
    
    # 将降维后的特征添加到 DataFrame
    reduced_df = pd.DataFrame(reduced_features, columns=[f'reduced_feature_{i}' for i in range(n_components)])
    
    df = pd.concat([df, reduced_df], axis=1)

    # 输出降维后的数据
    output_columns = ["user_id","total_counts","afternoon","evening","morning","night","text_feature","cluster", "reduced_feature_0", "reduced_feature_1"]
    df[output_columns].to_csv(output_path, index=False)

    # 可视化降维后的数据（如果指定了 image_output_path）
    if image_output_path:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=f"reduced_feature_0",
            y=f"reduced_feature_1",
            hue='cluster',
            data=df,
            palette=sns.color_palette("viridis", n_colors = len(df['cluster'].unique()))
        )
        plt.title(f"{method.upper()} Dimensionality Reduction (n_components={n_components})")
        plt.xlabel(f"Reduced Feature 1")
        plt.ylabel(f"Reduced Feature 2")
        plt.savefig(image_output_path, format='pdf')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform dimensionality reduction on clustered data.')
    parser.add_argument('data_path', type=str, help='Path to the input CSV file containing clustered data.')
    parser.add_argument('output_path', type=str, help='Path to the output CSV file for the reduced data.')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne'], help='Dimensionality reduction method (pca or tsne). Default is pca.')
    parser.add_argument('--n_components', type=int, default=2, help='Number of components to keep after reduction. Default is 2.')
    parser.add_argument('--image_output_path', type=str, default=None, help='Path to save the output image file. If not specified, no image is output.')
    args = parser.parse_args()

    dimensionality_reduction(
        data_path=args.data_path,
        output_path=args.output_path,
        method=args.method,
        n_components=args.n_components,
        image_output_path=args.image_output_path
    )