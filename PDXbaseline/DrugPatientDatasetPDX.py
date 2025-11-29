import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import ast


class DrugPatientDatasetPDX(Dataset):
    """
    PDX药物-患者组合的数据集类
    """

    def __init__(self, drug_file, patient_embed_file, sensitivity_file, model_file, ppi_file,
                 patient_feature_file, patient_name_file, tokenizer, max_length=512, num_classes=4):
        """
        初始化数据集

        Args:
            drug_file: 包含drug.id、smiles、treatment.type、afm、adj的CSV文件路径
            patient_embed_file: 包含所有患者特征的numpy文件路径 (num*1200*512)
            sensitivity_file: 包含model.id和mRECIST的CSV文件路径
            model_file: 包含model.id、patient.id、drug.id的CSV文件路径
            ppi_file: 包含所有患者蛋白质-蛋白质交互(PPI)矩阵的numpy文件路径 (1200*1200)
            patient_feature_file: 包含所有患者特征矩阵的numpy文件路径 (num*1200*1200)
            patient_name_file: 包含患者ID与索引映射的CSV文件路径
            tokenizer: 用于SMILES字符串的tokenizer
            max_length: SMILES字符串的最大长度
        """
        # 读取所有CSV文件
        drug_data = pd.read_csv(drug_file)
        sensitivity = pd.read_csv(sensitivity_file)
        model_mapping = pd.read_csv(model_file)

        # 读取患者名称映射文件
        patient_names = pd.read_csv(patient_name_file)

        # 创建patient.id到索引的映射字典
        patient_to_idx = {row['patient.id']: i for i, row in patient_names.iterrows()}

        try:
            patient_embeddings_array = np.load(patient_embed_file)
            print(f"Loaded patient embeddings with shape: {patient_embeddings_array.shape}")
        except Exception as e:
            print(f"Error loading patient embeddings: {e}")
            patient_embeddings_array = None

        try:
            ppi_matrices = np.load(ppi_file)
            print(f"Loaded PPI matrices with shape: {ppi_matrices.shape}")
        except Exception as e:
            print(f"Error loading PPI matrices: {e}")
            ppi_matrices = None

        try:
            patient_features_array = np.load(patient_feature_file)
            print(f"Loaded patient features with shape: {patient_features_array.shape}")
        except Exception as e:
            print(f"Error loading patient feature matrices: {e}")
            patient_features_array = None

        # 过滤掉标签为NA的数据
        sensitivity = sensitivity[sensitivity['mRECIST'] != 'NA']

        if num_classes == 2:
            label_mapping = {'CR': 0, 'PR': 0, 'SD': 0, 'PD': 1}
            sensitivity['label'] = sensitivity['mRECIST'].map(label_mapping)
        else:
            # 处理多分类标签
            label_mapping = {'CR': 0, 'PR': 1, 'SD': 2, 'PD': 3}
            sensitivity['label'] = sensitivity['mRECIST'].map(label_mapping)

        # 合并数据集
        merged_data = pd.merge(model_mapping, sensitivity, on='model.id')
        merged_data = pd.merge(merged_data, drug_data, on='drug.id')

        valid_rows = []
        for idx, row in merged_data.iterrows():
            patient_id = row['patient.id']
            if (patient_id in patient_to_idx and
                    patient_embeddings_array is not None and
                    patient_features_array is not None):
                valid_rows.append(idx)

        merged_data = merged_data.iloc[valid_rows].reset_index(drop=True)
        merged_data = merged_data.dropna(subset=['smiles', 'afm', 'adj', 'label', 'treatment.type'])
        merged_data.to_csv('filtered_data.csv', index=False)

        # 保存最终处理后的数据
        self.data = merged_data
        self.patient_to_idx = patient_to_idx
        self.patient_embeddings_array = patient_embeddings_array
        self.ppi_matrices = ppi_matrices
        self.patient_features_array = patient_features_array
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = row['patient.id']
        patient_idx = self.patient_to_idx[patient_id]

        # 处理AFM矩阵 (n*27)
        afm_matrix = self._process_matrix(row['afm'])

        # 处理ADJ矩阵 (n*3)
        adj_matrix = self._process_matrix(row['adj'])

        # 获取患者编码
        patient_encoding = self.patient_embeddings_array[patient_idx]

        # 获取患者特征矩阵
        patient_features = self.patient_features_array[patient_idx]

        # 获取PPI矩阵
        ppi_matrix = self.ppi_matrices

        # 获取标签
        label = row['label']

        # 处理SMILES字符串
        if row['treatment.type'] == 'single':
            # 单一药物处理
            smiles = row['smiles']
            encoding = self.tokenizer(
                smiles,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()

        elif row['treatment.type'] == 'combo':
            # 组合药物处理
            try:
                # 解析SMILES列表
                smiles_list = ast.literal_eval(row['smiles'])

                # 分别处理每个SMILES字符串
                encodings = [
                    self.tokenizer(
                        smiles,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    ) for smiles in smiles_list
                ]

                # 合并input_ids和attention_masks
                input_ids = sum(enc['input_ids'].squeeze() for enc in encodings)
                attention_mask = encodings[0]['attention_mask'].squeeze()

            except (SyntaxError, ValueError):
                # 如果解析失败，回退到单一药物处理
                print(f"无法解析组合药物SMILES: {row['smiles']}")
                smiles = row['smiles']
                encoding = self.tokenizer(
                    smiles,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].squeeze()
                attention_mask = encoding['attention_mask'].squeeze()

        else:
            # 处理未知的treatment.type
            raise ValueError(f"Unknown treatment type: {row['treatment.type']}")
        # 将张量转换为适当的形状
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'afm_features': torch.tensor(afm_matrix, dtype=torch.float32),
            'adj_features': torch.tensor(adj_matrix, dtype=torch.float32),
            'patient_encoding': torch.tensor(patient_encoding, dtype=torch.float32),  # 1200x512
            'patient_features': torch.tensor(patient_features, dtype=torch.float32),  # 1200x1200
            'ppi_matrix': torch.tensor(ppi_matrix, dtype=torch.float32),  # 1200x1200
            'label': torch.tensor(label, dtype=torch.long),
            'treatment_type': row['treatment.type']
        }

        return item

    def _process_matrix(self, matrix_str):
        """将字符串形式的矩阵转换为numpy数组"""
        try:
            matrix = ast.literal_eval(matrix_str)
            return np.array(matrix, dtype=np.float32)
        except (SyntaxError, ValueError):
            try:
                matrix = json.loads(matrix_str)
                return np.array(matrix, dtype=np.float32)
            except:
                print(f"无法解析矩阵字符串: {matrix_str[:50]}...")
                feature_dim = 27 if 'afm' in str(matrix_str) else 3
                return np.zeros((1, feature_dim), dtype=np.float32)


class DataCollatorPDX:
    """
    处理批量数据的整理器
    """

    def __call__(self, features):
        batch_size = len(features)
        if batch_size == 0:
            return {}

        # 提取特征
        labels = torch.stack([f['label'] for f in features])

        # 处理patient_encoding
        patient_encodings = torch.stack([f['patient_encoding'] for f in features])

        # 处理矩阵
        ppi_matrices = torch.stack([f['ppi_matrix'] for f in features])
        patient_features = torch.stack([f['patient_features'] for f in features])

        # 处理input_ids和attention_mask
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]

        # 处理AFM和ADJ特征
        afm_features = [f['afm_features'] for f in features]
        adj_features = [f['adj_features'] for f in features]

        # 对AFM特征进行填充
        max_afm_len = max(af.shape[0] for af in afm_features)
        padded_afm = []
        for af in afm_features:
            if af.shape[0] < max_afm_len:
                padding = torch.zeros((max_afm_len - af.shape[0], af.shape[1]), dtype=af.dtype)
                padded_afm.append(torch.cat([af, padding], dim=0))
            else:
                padded_afm.append(af)

        # 对ADJ特征进行填充
        max_adj_len = max(adj.shape[0] for adj in adj_features)
        padded_adj = []
        for adj in adj_features:
            if adj.shape[0] < max_adj_len:
                padding = torch.zeros((max_adj_len - adj.shape[0], adj.shape[1]), dtype=adj.dtype)
                padded_adj.append(torch.cat([adj, padding], dim=0))
            else:
                padded_adj.append(adj)

        # 对input_ids和attention_mask进行填充
        max_input_len = max(ids.shape[0] for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(input_ids, attention_mask):
            if ids.shape[0] < max_input_len:
                pad_width = max_input_len - ids.shape[0]
                padded_ids = torch.nn.functional.pad(ids, (0, pad_width), value=0)
                padded_mask = torch.nn.functional.pad(mask, (0, pad_width), value=0)
                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)
            else:
                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)

        # 创建批量
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "afm_features": torch.stack(padded_afm),
            "adj_features": torch.stack(padded_adj),
            "patient_encoding": patient_encodings,
            "ppi_matrix": ppi_matrices,
            "patient_features": patient_features,
            "labels": labels
        }

        return batch