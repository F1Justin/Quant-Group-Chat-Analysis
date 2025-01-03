# 时间序列与聚类的群聊分析

## 项目简介

本项目是数据量化推理课程的大作业，旨在分析群聊数据中的时间序列特征和用户行为模式。我们使用了以下技术手段：
- **时间序列分析**：对用户消息发送时间段的数据进行特征提取。
- **K-Means 聚类**：对用户的行为数据进行分组，识别行为模式。
- **降维与可视化**：使用主成分分析（PCA）和 t-SNE 技术展示聚类效果。

## 功能概述
1. **数据预处理**
    - 整合 CSV 文件，清洗无效用户数据。
    - 提取时间段活跃特征和文本特征。
    - 使用 TF-IDF 向量化文本数据。
2. **聚类分析**
    - 标准化数据，合并数值特征和文本特征。
    - 使用 K-Means 算法将用户分组。
    - 生成每个簇的统计信息。
3. **可视化**
    - 使用 PCA 和 t-SNE 进行降维与聚类效果展示。
    - 生成散点图和相关图表。

## 目录结构

```
├── data/                  # 存放原始数据的目录
├── results/               # 存放输出结果（如聚类结果 CSV 和图表）的目录
├── scripts/
│   ├── ...                #数据处理代码
├── report/
│   ├── report.tex         # 报告的 LaTeX 代码
│   ├── report.pdf         # 报告的 PDF 文件
│   ├── figures/           # 报告中引用的图片
├── README.md              # 项目说明文档
└── requirements.txt       # 依赖列表
```

## 快速开始

### 环境配置
1. 克隆项目：

     ```sh
     git clone https://github.com/F1Justin/Quant-Group-Chat-Analysis.git
     cd Quant-Group-Chat-Analysis
     ```

2. 安装依赖：

     ```sh
     pip install -r requirements.txt
     ```

### 数据处理与分析
1. 数据分析：

     ```sh
     python scripts/content_analysis.py
     ```

2. 聚类分析：

     ```sh
     python scripts/k-m-cluster.py
     ```

3. 降维与可视化：

     ```sh
     python scripts/dim_reduction.py
     ```

## 注意事项
1. **硬编码路径**
    本项目中部分路径和文件名是硬编码的，位于 `scripts/time_analysis.py` 和 `scripts/user_analysis.py` 文件中，需根据实际情况修改，例如：

     ```python
     DATA_DIR = "/path/to/data/"
     OUTPUT_DIR = "/path/to/output/"
     ```

2. **不建议直接使用**
    - 本项目代码仅供学习与参考，若需用于其他数据集，请确保：
      - 修改代码中的路径和特定参数。
      - 数据格式符合项目要求。
      - 在真实生产环境中，需增加异常处理和更灵活的配置管理。

3. **数据隐私**
    请确保使用的数据已脱敏或为公开数据，避免泄露用户隐私。

## 项目报告

项目详细技术与分析结果请参阅 `report/report.tex`。报告中描述了：
- 数据预处理的技术细节。
- 聚类算法的理论与实现。
- 降维可视化结果的分析与解读。

## 致谢

感谢数据量化推理课程提供的理论支持。
