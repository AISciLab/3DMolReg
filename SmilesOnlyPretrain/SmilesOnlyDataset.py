import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizerFast, TrainingArguments, Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import json
import ast
import os
from tokenizers import ByteLevelBPETokenizer

from SmilesOnlyPretrain.SmilesOnlyModel import SmilesOnlyModelConfig, SmilesOnlyModelForMaskedLM


class SmilesOnlyDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['smiles']

        # 处理AFM矩阵 (n*27)
        afm_matrix = self._process_matrix(row['afm'])

        # 处理ADJ矩阵 (n*3)
        adj_matrix = self._process_matrix(row['adj'])

        # 标记化SMILES字符串
        encoding = self.tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 将张量转换为适当的形状
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'afm_features': torch.tensor(afm_matrix, dtype=torch.float32),
            'adj_features': torch.tensor(adj_matrix, dtype=torch.float32),
        }

        return item

    def _process_matrix(self, matrix_str):
        """
        将字符串形式的矩阵转换为numpy数组
        """
        try:
            matrix = ast.literal_eval(matrix_str)
            return np.array(matrix, dtype=np.float32)
        except (SyntaxError, ValueError):
            try:
                matrix = json.loads(matrix_str)
                return np.array(matrix, dtype=np.float32)
            except json.JSONDecodeError:
                print(f"无法解析矩阵字符串: {matrix_str[:50]}...")
                feature_dim = 27 if 'afm' in matrix_str else 3
                return np.zeros((1, feature_dim), dtype=np.float32)


# 自定义数据整理器，处理批量数据
class MultimodalDataCollatorForMLM(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch_size = len(features)
        if batch_size == 0:
            return {}

        # 提取特征
        input_ids = torch.stack([f.pop("input_ids") for f in features])
        attention_mask = torch.stack([f.pop("attention_mask") for f in features])
        afm_features = [f.pop("afm_features") for f in features]
        adj_features = [f.pop("adj_features") for f in features]

        # 对AFM特征进行填充
        max_afm_len = max(af.shape[0] for af in afm_features)
        padded_afm = []
        for af in afm_features:
            if af.shape[0] < max_afm_len:
                padding = torch.zeros((max_afm_len - af.shape[0], af.shape[1]), dtype=af.dtype, device=af.device)
                padded_afm.append(torch.cat([af, padding], dim=0))
            else:
                padded_afm.append(af)

        # 对ADJ特征进行填充
        max_adj_len = max(adj.shape[0] for adj in adj_features)
        padded_adj = []
        for adj in adj_features:
            if adj.shape[0] < max_adj_len:
                padding = torch.zeros((max_adj_len - adj.shape[0], adj.shape[1]), dtype=adj.dtype, device=adj.device)
                padded_adj.append(torch.cat([adj, padding], dim=0))
            else:
                padded_adj.append(adj)

        # 创建批量
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "afm_features": torch.stack(padded_afm),
            "adj_features": torch.stack(padded_adj),
        }

        # 添加标签（用于MLM）
        if self.mlm:
            inputs, labels = self.torch_mask_tokens(batch["input_ids"].clone())
            batch["input_ids"] = inputs
            batch["labels"] = labels

        return batch


# 初始化配置和模型
def initialize_model(tokenizer):
    config = SmilesOnlyModelConfig.from_pretrained(
        'roberta-base',
        vocab_size=len(tokenizer),
        num_attention_heads=12,
        num_hidden_layers=6,
    )
    model = SmilesOnlyModelForMaskedLM(config)
    return model


# 创建并训练新的分词器
def train_new_tokenizer(train_files=None, output_dir="./custom_tokenizer"):
    """
    训练自定义分词器

    参数:
    train_files (list): 训练文本
    output_dir (str): 保存分词器

    返回:
    tokenizer: 训练好的分词器
    """
    # 如果未提供训练文件，则从CSV中提取SMILES字符串创建训练文件
    if train_files is None:
        # 从CSV文件中读取SMILES字符串
        df = pd.read_csv('model/out_embedding.csv')
        smiles_list = df['smiles'].tolist()

        # 创建包含所有SMILES字符串的临时文件
        os.makedirs("temp", exist_ok=True)
        train_file = "temp/smiles_corpus.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for smiles in smiles_list:
                f.write(f"{smiles}\n")

        train_files = [train_file]

    # 创建分词器实例
    tokenizer = ByteLevelBPETokenizer()

    # 训练分词器
    tokenizer.train(
        files=train_files,
        vocab_size=512,  # 词汇表大小
        min_frequency=3,  # 最小词频
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    # 保存分词器
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)

    # 加载为RobertaTokenizerFast
    fast_tokenizer = RobertaTokenizerFast.from_pretrained(output_dir)
    # 分词器指定掩码token
    fast_tokenizer.mask_token = "<mask>"

    return fast_tokenizer


# 训练模型
def train_model(model, train_dataset, eval_dataset=None, tokenizer=None, output_dir="model/smilesOnly_roberta_model"):
    """
    训练代码

    参数:
    train_dataset (list): 训练数据集
    eval_dataset (list): 评估数据集
    tokenizer (RobertaTokenizerFast): 分词器
    output_dir (str): 保存目录

    """

    # 创建数据整理器
    data_collator = MultimodalDataCollatorForMLM(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=128,
        save_steps=20_000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        logging_dir='./logs',
        logging_steps=500,
        save_strategy="steps",
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 开始训练
    trainer.train()

    # 手动保存模型和分词器，使用safe_serialization=False
    print("保存模型...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    print(f"模型已保存到 {output_dir}")

    return model, trainer


# 主函数
def main():
    # 训练自定义分词器
    print("开始训练自定义分词器...")
    tokenizer = train_new_tokenizer(
        train_files= None,  # None表示从CSV中提取SMILES字符串
        output_dir="model/Mulit_roberta_tokenizer"
    )
    print("分词器训练完成！")

    # 创建数据集
    train_dataset = SmilesOnlyDataset(
        csv_path='model/out_embedding_train.csv',
        tokenizer=tokenizer,
        max_length=512,
    )

    eval_dataset = SmilesOnlyDataset(
        csv_path='model/out_embedding_eval.csv',
        tokenizer=tokenizer,
        max_length=512,
    )

    # 初始化模型
    model = initialize_model(tokenizer)

    # 训练模型
    model, trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        output_dir = "model/Mulit_roberta_model"
    )

    print("模型训练完成！")


if __name__ == "__main__":
    main()