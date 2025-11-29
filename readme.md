# MultiSmilesModel: Multimodal Drug Response Prediction

[This project proposes a deep learning model based on multimodal SMILE sequences and patient characteristics to predict drug response.]

## 🛠 环境安装 (Installation)

为了确保代码能够正常运行，建议创建一个独立的 Conda 环境以避免依赖冲突。

### 1. 创建并激活 Conda 环境
推荐使用 Python 3.8 或更高版本：

```bash
# 创建名为 multismiles 的环境
conda create -n MultiPharma python=3.8

# 激活环境
conda activate MultiPharma
```

### 2. 安装依赖库
本项目基于 PyTorch 和 HuggingFace Transformers 构建。请先根据您的 CUDA 版本安装 PyTorch（参考 [PyTorch官网](https://pytorch.org/)），然后安装其他核心依赖。

```bash
# 1. 安装 PyTorch (以 CUDA 11.8 为例，请根据您的显卡驱动实际情况调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. 安装核心依赖库
pip install transformers wandb scikit-learn pandas numpy tqdm
```

---

## 📂 项目结构 (Project Structure)

为了确保代码能直接使用默认参数运行，建议保持以下目录结构：

```text
Project/
├── main.py                # 包含所有任务的启动逻辑
├── model/                 # 存放预训练模型权重和分词器
│   ├── multiSmiles_model/
│   │   └── checkpoint-80000/
│   └── custom_tokenizer/
├── data/                  # 数据文件目录
│   ├── PDTC/              # PDTC 相关数据 (.csv, .npy)
│   ├── PDX/               # PDX 相关数据
│   └── TCGA/              # TCGA 相关数据
└── README.md
```

---

## 🚀 使用方法 (Usage)

本项目使用统一的 `main.py` 入口，通过 **子命令 (Subcommands)** 来区分不同的实验任务。

代码设计支持两种运行模式：
1.  **快速复现模式**：不传递任何参数，代码将自动加载代码中预设的默认路径和超参数。
2.  **自定义实验模式**：通过命令行参数覆盖默认配置。

### 1. PDTC 回归任务 (PDTC Regression)
运行基础的 PDTC 数据集训练任务。默认设置下，将使用Sample数据进行训练，预测Model数据结果

*   **默认运行:**
    ```bash
    python main.py pdtc-reg
    ```

*   **自定义参数运行:**
    ```bash
    # 修改批次大小为 64，学习率为 5e-5，仅重复运行 3 次
    python main.py pdtc-reg \
        --batch_size 64 \
        --learning_rate 5e-5 \
        --repeat_times 3
    ```

### 2. PDTC 10折交叉验证 (PDTC 10-Fold CV)
运行严格的 10 折交叉验证实验，用于评估模型泛化能力。

*   **默认运行:**
    ```bash
    python main.py pdtc-ten
    ```

*   **自定义参数运行:**
    ```bash
    # 指定 WandB 项目名称，增加 Epoch 数
    python main.py pdtc-ten \
        --project_name "PDTC_10Fold_Exp1" \
        --num_epochs 200 \
        --batch_size 128
    ```

### 3. PDX 微调任务 (PDX Finetune)
在 PDX 数据集上进行微调训练。

*   **默认运行:**
    ```bash
    python main.py pdx-fine
    ```

*   **指定特定数据路径:**
    ```bash
    # 如果您有新的数据文件，可以直接通过参数指定
    python main.py pdx-fine \
        --drug_file "./data/New_PDX/drugs.csv" \
        --sample_sensitivity_file "./data/New_PDX/response.csv" \
        --learning_rate 2e-5
    ```

### 4. TCGA 预测任务 (TCGA Prediction)
使用训练好的模型对 TCGA 队列数据进行推理预测。

*   **默认运行:**
    ```bash
    python main.py tcga
    ```

*   **指定输出文件:**
    ```bash
    python main.py tcga \
        --tcga_output_file "./results/tcga_final_predictions.csv"
    ```
    
### 4. 预训练任务 (Pretrain)
如果希望重新预训练模型，可是使用此任务

*   **默认运行:**
    ```bash
    python main.py Pretrain
    ```

*   **指定预训练样本:**
    ```bash
    python main.py tcga \
        --train_data 'data/Pretrain/out_embedding_train.csv'
    ```

---

## ⚙️ 常用参数说明 (Arguments)

可以通过 `python main.py <task> --help` 查看每个任务支持的完整参数列表。以下为通用参数：

| 参数名 | 类型 | 说明 | 默认值 (Default)                   |
| :--- | :--- | :--- |:--------------------------------|
| `--model_path` | str | 预训练模型路径 | `./model/multiSmiles_model/...` |
| `--tokenizer_path` | str | 分词器路径 | `./model/custom_tokenizer`      |
| `--batch_size` | int | 训练批次大小 | 128                     |
| `--learning_rate` | float | 初始学习率 | 1e-4 或 3e-5                     |
| `--num_epochs` | int | 训练轮数 | 150 (PDX默认为50)                  |
| `--dropout_rate` | float | Dropout 概率 | 0.0 或 0.2                       |
| `--project_name` | str | WandB 项目名称 | (根据任务自动命名)                      |
| `--weight_decay` | float | 权重衰减系数 | 0.0001 / 0.001                  |

---

## 📄 引用 (Citation)

如果您在研究中使用了本项目，请引用以下文献：

```bibtex

```
```