# 🌾 Dry Bean Dataset Premium Machine Learning Pipeline
> **干豆多算法高精度分类与学术级机器学习工程集成系统**

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

本仓库是一个基于 **Dry Bean Dataset**（干豆形状与几何特征数据集）构建的全流程、工业级机器学习工程项目。本项目在满足高校课程考核要求的基础上，深度集成了课堂所讲的核心知识点（如非线性激活升维、Batch Normalization、Dropout 抑制过拟合、验证集早停、超参数网格搜索、后处理置信度过滤、多噪声强度鲁棒性评测等），实现了一键式端到端流水线运行。

---

## 🎨 系统核心亮点

### 1. 🛡️ 工业级数据清洗与无损几何重建 (Data Cleaning & Reconstruction)
- **脏数据纠错与清洗**：正则化修复损坏标签（如 `D3RMAS0N`、`B0MBAY` 等 25 种脏标签变体自动归并至 7 类标准标签），利用正则剔除特征中的非法物理后缀（如 ` cm`、`?` 等）。
- **数学无损几何重建**：拒绝简单粗暴的中位数填充。利用干豆的几何先验公式（如 $Solidity = \frac{Area}{ConvexArea}$，周长 $Perimeter = \sqrt{\frac{4 \pi \cdot Area}{roundness}}$）对缺失值进行 **100% 精确数学还原**，保留自然分布特征。
- **数据防泄露与去重**：自动剔除训练集中的重复样本，并在划分时深度检测并去除验证集/测试集与训练集之间的信息泄露（Data Leakage）。
- **3-Sigma 离群点截断**：应用拉依达准则（3-Sigma）对异常录入点（如负面积值等）进行截断与规范化。

### 2. 📉 高维特征冗余降维 (PCA Feature Reduction)
- **多重共线性消除**：EDA 分析指出特征间高度冗余（如 `Area` 与 `ConvexArea` 相关系数达 `0.9999`）。
- **主成分分析**：通过方差贡献率分析，利用 **10个主成分（PC1~PC10）** 保留了原始数据 **99.95%** 的方差能量，在大幅降低维度与计算复杂度的同时保留完整特征信息。

### 3. 🧠 3层前馈深度神经网络 (3-Layer Deep ANN)
- **科学网络拓扑**：设计并实现 `Input(10D) -> 128 (BatchNorm -> ReLU -> Dropout) -> 64 (BatchNorm -> ReLU -> Dropout) -> 32 (BatchNorm -> ReLU) -> Output(7D)` 网络。
- **批归一化 (BatchNorm)**：引入层批归一化平滑损失曲面，大幅加速收敛。
- **随机失活 (Dropout)**：设置 `Dropout(0.3)` 限制模型对单一特征的依赖。
- **早停机制 (Early Stopping)**：基于验证集 Loss 进行监控（`patience=12`），在触发过拟合前自动截断训练并还原历史最优权重。
- **置信度拒绝过滤器 (Softmax Confidence Filter)**：后处理阶段引入置信阈值限制（默认 `0.4`）。若分类置信度低于阈值，则拒绝做出预测，输出 `-1` 标记转人工复检，防范工业分拣中的严重误判。

### 4. 🧪 15档噪声压力测试与鲁棒性评估 (Robustness Stress Test)
- 构建了高斯噪声（传感器漂移）、椒盐噪声（信号翻转）和特征屏蔽（传感器损坏）等 3 大维度、15 档噪声梯度的自动压力测试。

---

## 🏗️ 模块化项目结构

```text
期末作业/
├── data/                            # 原始数据集文件夹
│   ├── Dry_Bean_Dataset_Dirty_train.csv
│   ├── Dry_Bean_Dataset_Dirty_val.csv
│   └── Dry_Bean_Dataset_Dirty_test.csv
├── core/                            # 核心工程代码模块
│   ├── dataloader.py                # 标签纠错、物理公式重建缺失值及 PCA 降维模块
│   ├── models.py                    # SVM、随机森林(RF)、三层深度神经网络(ANN)定义
│   └── evaluator.py                 # 分类指标、吞吐速度(FPS)及 15 档噪声鲁棒性测试模块
├── results/                         # 自动生成的成果输出文件夹
│   ├── results.json                 # 实验全量指标数据库 (JSON 序列化)
│   ├── experiment_report.md         # 自动格式化的对比报告 Markdown 表格
│   ├── overfitting_comparison.png   # 训练集 vs 测试集过拟合对比图
│   ├── speed_comparison.png         # 推理吞吐率 (Log-scale FPS) 柱状图
│   ├── robustness_comparison.png    # 高斯/椒盐/屏蔽多强度噪声精度衰减折线图
│   ├── ann_loss_curve.png           # ANN 训练/验证 Loss 双收敛曲线图
│   └── confusion_matrix_[MODEL].png # 混淆矩阵热力图（包含计数与归一化百分比）
├── analysis/                        # 数据分析与可视化绘图脚本
│   ├── eda.py                       # 深度探索性数据分析 (生成特征KDE、箱线图、聚类投影图)
│   └── plot_results.py              # 数据驱动动态绘图脚本 (完全读取 results.json 绘图)
├── main.py                          # 统一命令行调度入口 (CLI Pipeline)
├── requirements.txt                 # 环境依赖配置文件
└── README.md                        # 本说明文件
```

---

## 📊 实验多维指标对比结果

以下为系统在测试集上自动跑出的真实实验对比数据：

| 指标维度 | 支持向量机 (SVM) | 随机森林 (Random Forest) | 神经网络 (ANN) |
| :--- | :---: | :---: | :---: |
| **最优超参数** | `C=10.0`, `kernel=rbf` | `n_estimators=150`, `max_depth=12` | `lr=0.001`, `dropout_prob=0.3` |
| **测试集 Accuracy** | **93.56%** | 92.50% | **93.60%** |
| **测试集 Precision** | 93.58% | 92.57% | 93.71% |
| **测试集 Recall** | 93.56% | 92.50% | 93.60% |
| **测试集 F1-score** | 93.57% | 92.52% | **93.65%** |
| **分类拒绝率 (ANN独有)** | N/A | N/A | **0.11%** (3个样本) |
| **置信样本准确率** | N/A | N/A | **93.70%** |
| **单样本推理耗时** | 0.0987 ms | 0.0143 ms | **0.00036 ms** |
| **推理吞吐量 (FPS)** | 10,136.3 | 70,115.7 | **2,747,365.1** |
| **训练集 Accuracy** | 93.72% | 98.45% | 93.43% |
| **过拟合差值 (绝对差)** | **0.0016** (极佳) | 0.0595 | **-0.0017** (完全无过拟合) |
| **过拟合风险评估** | **极低 (Low)** | **低 (Low)** | **极低 (Low)** |

### 🔍 核心实验结论
1. **垃圾进，垃圾出 (GIGO) 的验证**：通过彻底的拼写归并、缺失值公式无损重建，各模型测试集精度从原本的 **~89% 大幅跃升至 93.5% 以上**。
2. **神经网络 (ANN) 卓越的泛化表现**：在 Dropout、BatchNorm 与早停的加持下，神经网络的过拟合差值被限制在负值（$-0.17\%$），在测试集上取得了最优的准确率（**93.60%**）与 F1-score（**93.65%**）。
3. **极速推理吞吐量**：基于矩阵计算优化的 PyTorch ANN 在测试中跑出了 **274.7 万 FPS** 的惊人吞吐率，展现出极高的工业分拣应用价值。
4. **集成学习的超强抗噪能力**：在椒盐噪声压力测试中，SVM 在 10% 强度下精度即发生暴跌，而随机森林（RF）凭借 Bagging 多数投票机制，在 10% 椒盐噪点强度下依然稳健地保持了 **82.46%** 的准确率。

---

## 🛠️ 如何快速运行

### 1. 安装环境依赖
```bash
pip install -r requirements.txt
```

### 2. 执行数据探索 (EDA) 分析
运行此脚本将在 `results/` 下生成特征 KDE 对比、相关性热力图、分组箱线图、以及 PCA/t-SNE 2D 降维空间聚类投影图：
```bash
python analysis/eda.py
```

### 3. 一键运行完整机器学习流水线 (Pipeline)
顺序训练和评估 SVM、ANN、随机森林，保存全量指标并自动输出精美的可视化结果：
```bash
python main.py --model all
```

### 4. 高阶参数说明
- `--tune`：在验证集上执行超参数网格搜索。
- `--train_noise`：在加噪后的训练集上对模型重新训练，以此测试训练时噪声增强的鲁棒性影响（默认关闭）。
- `--no_noise`：禁用测试集上的 15 档噪声梯度测试以加快运行速度。
- `--pca_components COMPONENTS`：主成分降维维度，输入 `-1` 禁用 PCA 并保留全部 16 维特征。

---

## 📄 自动生成的交付物

项目运行后，所有学术成果和可视化图表都会静默保存在 `results/` 文件夹下，您可以直接复制使用：
- `experiment_report.md`：自动排版好的 Markdown 格式实验数据对比表格。
- `期末大论文写作大纲框架.md`：为您量身定制的学位大论文目录与内容撰写指南，可直接导入您的最终论文。
- `ann_loss_curve.png` / `robustness_comparison.png` 等 10 余张精美的分析与对比图表。
