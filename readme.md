***

# 3DMolReg: Multimodal Drug Response Prediction

Identification of drug response in cancer patients by using deep learning has become the foundation of precision medicine.
Unfortunately, patients with drug response are often scarce, limiting the performance of deep learning.
Simultaneously, most methods use separate models for sequence or three-dimensional (3D) structures, hindering fine-grained interaction cross modalities. 
Furthermore, the multi-scale regulatory networks in life system play the key role in patient response to drugs.
Therefore, we propose a deep learning framework, 3DMolReg, that designs a multimodal molecular language model and patient representation learning to improve clinical drug response prediction. In the 3D structure-aware multimodal language model, molecular conformations are discretized into tokens as context and prompt knowledge for sequence masking models, enabling the fine-grained interaction cross modalities. In patient representation learning, regulatory networks at gene, protein and pathway levels are hierarchically mapped into Transformer by adaptive functions. Across multiple scenarios of drug response and disease progression prediction, 3DMolReg achieved superior performance and captured the pharmacology and biochemistry mechanisms. Importantly, the sensitive drugs predicted by 3DMolReg can improve the survival outcomes of clinical patients, further suggesting its potential in anticancer drug therapies.

## 🛠 Installation

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

To ensure the code runs seamlessly with default parameters, it is recommended to maintain the following directory structure:

```text
Project/
├── main.py                # Entry point containing launch logic for all tasks
├── model/                 # Stores pretrained model weights and tokenizer
│   ├── multiSmiles_model/
│   │   └── checkpoint-80000/
│   └── custom_tokenizer/
├── data/                  # Data directory
│   ├── PDTC/              # PDTC related data (.csv, .npy)
│   └── PDX/               # PDX related data (.csv, .npy)
└── README.md
```

---

## 🚀 Usage

This project uses a unified `main.py` entry point and utilizes **Subcommands** to distinguish between different experimental tasks.

The code is designed to support two execution modes:
1.  **Reproduction Mode:** Run without arguments to automatically load the preset default paths and hyperparameters defined in the code.
2.  **Custom Experiment Mode:** Override default configurations via command-line arguments.

### 1. Pretraining
Use this task if you wish to re-pretrain the model.

*   **Default Execution:**
    ```bash
    python main.py Pretrain
    ```

*   **Specify Pretraining Data:**
    ```bash
    python main.py Pretrain \
        --train_data 'data/Pretrain/out_embedding_train.csv'
    ```

### 2. TransferEvaluation Task
Runs the baseline PDTC dataset training task. By default, it trains using `Sample` data and predicts outcomes on `Model` data.

*   **Default Execution:**
    ```bash
    python main.py pdtc-reg
    ```

*   **Custom Execution:**
    ```bash
    # Set batch size to 64, learning rate to 5e-5, and repeat 3 times
    python main.py pdtc-reg \
        --batch_size 64 \
        --learning_rate 5e-5 \
        --repeat_times 3
    ```

### 3. PDTC CrossValidation
Runs a rigorous 10-fold cross-validation experiment to evaluate the model's generalization capability.

*   **Default Execution:**
    ```bash
    python main.py pdtc-ten
    ```

*   **Custom Execution:**
    ```bash
    # Specify WandB project name and increase number of Epochs
    python main.py pdtc-ten \
        --project_name "PDTC_10Fold_Exp1" \
        --num_epochs 200 \
        --batch_size 128
    ```
    
### 4. PDX CrossValidation
Runs a rigorous 10-fold cross-validation experiment to evaluate the model's generalization capability.

*   **Default Execution:**
    ```bash
    python main.py pdx-clas
    ```

*   **Custom Execution:**
    ```bash
    # Specify WandB project name and increase number of Epochs
    python main.py pdx-clas \
        --project_name "PDX_10Fold_Exp1" \
        --num_epochs 200 \
        --batch_size 128
    ```

---

## ⚙️ Arguments

You can view the full list of supported parameters for any task by running `python main.py <task> --help`. Below are the common arguments:

| Argument | Type | Description | Default Value                       |
| :--- | :--- | :--- |:------------------------------------|
| `--model_path` | str | Path to the pretrained model | `./model/multiSmiles_model/...`     |
| `--tokenizer_path` | str | Path to the tokenizer | `./model/custom_tokenizer`          |
| `--batch_size` | int | Training batch size | 128                                 |
| `--learning_rate` | float | Initial learning rate | 1e-4 or 3e-5                        |
| `--num_epochs` | int | Number of training epochs | 150                                 |
| `--dropout_rate` | float | Dropout probability |                                     |
| `--project_name` | str | WandB project name | (Named automatically based on task) |
| `--weight_decay` | float | Weight decay coefficient | 0.0001 / 0.001                      |

---

## 📄 Citation

If you use this project in your research, please cite the following reference:

```bibtex

```




