import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 默认黑体
plt.rcParams['axes.unicode_minus'] = False

# 自动确定项目路径
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
if os.path.basename(current_dir) == 'analysis':
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

results_dir = os.path.join(project_root, 'results')
json_path = os.path.join(results_dir, 'results.json')

def load_results():
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到结果数据文件: {json_path}。请先运行 main.py 生成结果。")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_overfitting(results):
    models = list(results.keys())
    train_acc = [results[m]['overfitting']['Train Acc'] for m in models]
    test_acc = [results[m]['overfitting']['Test Acc'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, train_acc, width, label='训练集准确率 (Train Acc)', color='#3498db')
    rects2 = ax.bar(x + width/2, test_acc, width, label='测试集准确率 (Test Acc)', color='#e74c3c')

    ax.set_ylabel('Accuracy')
    ax.set_title('各模型训练集与测试集精度对比（过拟合分析）')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower left')
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    # 标注数值
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    save_path = os.path.join(results_dir, 'overfitting_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"过拟合对比图已保存至: {save_path}")

def plot_fps(results):
    models = list(results.keys())
    fps_vals = [results[m]['speed']['FPS'] for m in models]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # 使用渐变紫色系
    colors = ['#9b59b6', '#8e44ad', '#6c5ce7'][:len(models)]
    bars = ax.bar(models, fps_vals, color=colors, width=0.5)
    ax.set_ylabel('推理速度 FPS (对数坐标)')
    ax.set_title('各模型每秒推理样本数对比 (FPS Comparison)')
    ax.set_yscale('log') # 神经网络预测太快，使用对数坐标
    ax.grid(axis='y', linestyle=':', alpha=0.6, which="both")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, weight='bold')

    fig.tight_layout()
    save_path = os.path.join(results_dir, 'speed_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"推理速度对比图已保存至: {save_path}")

def plot_robustness_gradients(results):
    models = list(results.keys())
    
    # 检查是否有噪声数据
    if 'robustness_gradients' not in results[models[0]]:
        print("未在 results.json 中找到噪声梯度数据，跳过抗噪折线图绘制。")
        return
        
    noise_types = ['gaussian', 'salt_pepper', 'feature_dropout']
    noise_titles = {
        'gaussian': '高斯噪声抗性曲线 (Gaussian Noise)',
        'salt_pepper': '椒盐噪声抗性曲线 (Salt & Pepper Noise)',
        'feature_dropout': '特征屏蔽抗性曲线 (Feature Dropout)'
    }
    x_labels = {
        'gaussian': '标准差 Standard Deviation (std)',
        'salt_pepper': '噪点比例 Noise Probability',
        'feature_dropout': '屏蔽概率 Dropout Probability'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'SVM': '#2ecc71', 'ANN': '#e67e22', 'RF': '#9b59b6'}
    markers = {'SVM': 'o', 'ANN': 's', 'RF': '^'}
    
    for idx, nt in enumerate(noise_types):
        ax = axes[idx]
        for m in models:
            if 'robustness_gradients' in results[m] and nt in results[m]['robustness_gradients']:
                grad_data = results[m]['robustness_gradients'][nt]
                intensities = [item['intensity'] for item in grad_data]
                accuracies = [item['accuracy'] for item in grad_data]
                
                # 绘制折线
                ax.plot(intensities, accuracies, label=m, color=colors.get(m, '#95a5a6'),
                        marker=markers.get(m, 'x'), linewidth=2, markersize=6)
                
        ax.set_title(noise_titles[nt], fontsize=12, weight='bold')
        ax.set_xlabel(x_labels[nt])
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'robustness_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"抗噪鲁棒性折线图已保存至: {save_path}")

def plot_ann_loss_curve(results):
    if 'ANN' not in results or 'loss_history' not in results['ANN']:
        print("未在 results.json 中找到 ANN 的 Loss 训练历史，跳过 Loss 曲线绘制。")
        return
        
    loss_history = results['ANN']['loss_history']
    train_loss = loss_history.get('train', [])
    val_loss = loss_history.get('val', [])
    
    if not train_loss:
        return
        
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='#1abc9c', linewidth=2)
    if val_loss:
        plt.plot(epochs, val_loss, label='Val Loss', color='#e67e22', linewidth=2, linestyle='--')
        
    plt.title('人工神经网络 (ANN) 训练与验证 Loss 收敛曲线', fontsize=12, weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'ann_loss_curve.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ANN Loss 曲线图已保存至: {save_path}")

def plot_confusion_matrices(results):
    models = list(results.keys())
    
    # 干豆类别名称 (标准化顺序)
    class_names = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']
    
    for m in models:
        if 'confusion_matrix' in results[m] and results[m]['confusion_matrix']:
            cm = np.array(results[m]['confusion_matrix'])
            
            # 类别可能少于 7 个（如果测试样本太少），通常按实际大小对齐
            num_classes = cm.shape[0]
            labels = class_names[:num_classes]
            
            plt.figure(figsize=(10, 8))
            
            # 归一化以显示百分比（安全除法，防止除以零的警告）
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.divide(
                cm.astype('float'), 
                row_sums, 
                out=np.zeros_like(cm, dtype=float), 
                where=row_sums != 0
            )
            
            # 使用更优雅的 Blues/Oranges 渐变色
            cmap_choice = 'Blues' if m == 'SVM' else ('Oranges' if m == 'ANN' else 'Purples')
            
            # 创建热力图，同时显示数量和比例
            annot_labels = np.empty_like(cm, dtype=object)
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    annot_labels[r, c] = f"{cm[r, c]}\n({cm_normalized[r, c]*100:.1f}%)"
                    
            sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap=cmap_choice,
                        xticklabels=labels, yticklabels=labels, square=True, cbar=True,
                        linewidths=0.5, annot_kws={"fontsize": 9})
            
            plt.title(f'{m} 模型分类混淆矩阵热力图 (Confusion Matrix)', fontsize=12, weight='bold')
            plt.ylabel('True Class (真实类别)')
            plt.xlabel('Predicted Class (预测类别)')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            save_path = os.path.join(results_dir, f'confusion_matrix_{m}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"{m} 混淆矩阵热力图已保存至: {save_path}")

if __name__ == "__main__":
    print("=== Start Dynamic Result Plotting ===")
    try:
        results = load_results()
        plot_overfitting(results)
        plot_fps(results)
        plot_robustness_gradients(results)
        plot_ann_loss_curve(results)
        plot_confusion_matrices(results)
        print("=== All visual comparison charts updated successfully! ===")
    except Exception as e:
        print(f"Error occurred during plotting: {e}")
