<h1 align="center">基于SADA的教育对话文本新意图发现</h1>

<div align="center">

**[中文](README.md) | [English](README_EN.md)**

</div>

---

## 项目摘要

理解师生对话中的教育意图对于发现有效的教学模式和优化教学策略至关重要。传统的分析方法依赖于静态、预定义的编码方案，但这些方案难以适应多样化的教育场景，且手动优化以捕捉新的教育意图需要昂贵的专家成本。

尽管数据驱动技术对于从大规模未标注语料库中自动发现新教育意图具有重要意义，但这一方向的研究仍相对不足。

因此，本项目提出了**新教育意图发现** （NEID）方法。我们设计了一种新颖的迁移学习框架，将两阶段新意图发现 (NID) 方法扩展到教育领域当中，并增强了语义感知判别表示能力。

## 项目架构

1. 阶段1：我们在公共数据集和任务数据集上进行多任务预训练。
2. 阶段2：我们进行知识蒸馏，在教师模型的指导下，学生模型内化噪声检测能力。
3. 阶段3：我们进行基于SADA增强的最近邻对比学习。训练完成后，我们采用一种简单的非参数聚类算法来获得聚类结果，其中每个聚类表示一种特定的教育意图。


### 项目结构

```
.
├── main.py                   # 主程序入口
├── sada.py                   # 对比学习与最近邻实现
├── mtp.py                    # 多任务预训练
├── intent_pretrain.py        # 意图预训练管理器
├── methods.py                # 知识蒸馏与SADA方法
├── model.py                  # 模型架构定义
├── dataloader.py             # 数据加载与处理
├── init_parameter.py         # 参数初始化
├── requirements.txt          # 依赖
│
├── scripts/
│   └── SADA_talkmoves.sh     # 运行脚本
│
├── utils/
│   ├── tools.py              # 工具函数
│   ├── memory.py             # 记忆银行实现
│   ├── contrastive.py        # 对比学习组件
│   ├── build_ml.py           # 模型构建工具
│   ├── adamW.py              # AdamW优化器
│   ├── neighbor_dataset.py   # 邻居数据集
│   └── sequence_classification.py  # 序列分类
│
├── data/                     # 内部数据集
│   └── talkmoves/            
│       ├── train.tsv
│       ├── dev.tsv
│       └── test.tsv
│
├── data_external/            # 外部数据集
│   └── clinc/                
│       └── dataset.json
│
```

---

## 项目实现

### 安装依赖

```bash
pip install -r requirements.txt
```

### 创建模型目录

**重要**：请创建 `model` 文件夹用于存放下载的预训练模型：

```bash
mkdir -p model/bert-base-uncased
```

### 运行命令

```bash
bash SADA_talkmoves.sh 0,1,2,3
```

其中 `0,1,2,3` 指定使用的 GPU 编号（可多卡运行）。

### 支持的参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | GPU 设备 ID | `0,1,2,3` |
| `internal_dataset` | 内部数据集 | `talkmoves` |
| `labeled_ratio` | 标注数据比例 | `0.1`, `0.25` |
| `known_cls_ratio` | 已知类别比例 | `0.5`, `0.75` |
| `input_strategy` | 输入策略 | `CONTEXT`, `ORIGINAL` |
| `with_speaker` | 是否使用说话者信息 | `0`, `1` |
| `view_strategy` | 视角策略 | `SADA`, `rtr`, `shuffle`, `none` |


### 输出结果

运行结果将保存在：
- **模型检查点**: `save_models/checkpoints/`
- **训练日志**: `logs/`
- **可视化结果**: `save_models/` (UMAP可视化)
- **评估结果**: `results.csv`

---


本项目部分代码参考和改编自以下开源项目：

1. GitHub: https://github.com/zhang-yu-wei/MTP-SADA
2. GitHub: https://github.com/WenxiongLiao/mask-bert