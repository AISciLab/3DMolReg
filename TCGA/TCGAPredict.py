import json
import os
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split, Subset
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, auc, \
    precision_recall_curve, mean_squared_error, mean_absolute_error, r2_score
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from torch.amp import autocast
from torch.utils.data import ConcatDataset

from TCGA.DrugPatientClassifier import DrugPatientClassifier
from TCGA.DrugPatientDatasetTCGA import DataCollatorTCGA, DrugPatientDatasetTCGA
from PDTCbaseline.DrugPatientDatasetPDTC import DrugPatientDatasetPDTC, DataCollatorPDTC

import warnings
from adabelief_pytorch import AdaBelief
from scipy import stats

warnings.simplefilter("ignore", category=FutureWarning)


def calculate_regression_metrics(y_true, y_pred):
    """
    计算多个回归评价指标

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        包含多个指标的元组
    """
    # 将数据转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算MSE和RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # 计算EVS (Explained Variance Score)
    evs = 1 - np.var(y_true - y_pred) / np.var(y_true)

    # 计算MedAE (Median Absolute Error)
    medae = np.median(np.abs(y_true - y_pred))

    # 计算MaxAE (Maximum Absolute Error)
    maxae = np.max(np.abs(y_true - y_pred))

    # 计算MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 计算相关系数
    pearson_corr, _ = stats.pearsonr(y_true, y_pred)
    spearman_corr, _ = stats.spearmanr(y_true, y_pred)

    # 计算MAE和R2
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    return {
        'mse': mse,
        'rmse': rmse,
        'evs': evs,
        'medae': medae,
        'maxae': maxae,
        'mape': mape,
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'mae': mae,
        'r2': r2
    }


def train_classification_model(model, train_dataset, val_dataset, batch_size=16, num_epochs=5, learning_rate=2e-5,
                               device='cuda', weight_decay=0.01, warmup_steps=0,
                               project_name="drug_patient_regression", run_name="run_1",
                               use_mixed_precision=True, gradient_accumulation_steps=1,
                               num_classes=4):
    """
    训练分类模型，支持混合精度训练

    参数:
        model: 组合模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        weight_decay: 权重衰减
        warmup_steps: 预热步数
        project_name: wandb项目名称
        run_name: wandb运行名称
        use_mixed_precision: 是否使用混合精度训练
        gradient_accumulation_steps: 梯度累积步数
    """
    # 记录超参数
    wandb.config.update({
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "model": model.__class__.__name__,
        "optimizer": "AdamW",
        "mixed_precision": use_mixed_precision
    })

    # 将模型移至设备
    model = model.to(device)

    # 创建数据加载器
    data_collator = DataCollatorPDTC()
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=data_collator, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.drug_model.named_parameters() if p.requires_grad],
            "lr": learning_rate,
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.patient_encoder.named_parameters()],
            "lr": learning_rate,
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.classifier.named_parameters()],
            "lr": learning_rate,
            "weight_decay": weight_decay
        }
    ]

    optimizer = AdaBelief(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=1e-16,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        weight_decouple=True,
        rectify=True,
        print_change_log=False
    )
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # 设置混合精度训练
    if use_mixed_precision and device != 'cpu':
        from torch.amp import GradScaler  # 推荐用法
        scaler = GradScaler()  # 指定设备
    else:
        scaler = None

    # 添加早停机制
    best_val_loss = float('inf')
    train_targets, train_predictions = [], []

    # 训练循环
    best_val_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # 训练循环
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        step = 0

        for batch in progress_bar:
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            afm_features = batch['afm_features'].to(device)
            adj_features = batch['adj_features'].to(device)
            patient_encoding = batch['patient_encoding'].to(device)
            patient_features = batch['patient_features'].to(device)
            PPI_matrix = batch['ppi_matrix'].to(device)
            labels = batch['labels'].to(device)

            # 清除梯度
            optimizer.zero_grad()

            # 使用混合精度训练
            if use_mixed_precision and device != 'cpu':
                with autocast("cuda"):
                    # 前向传播
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        afm_features=afm_features,
                        adj_features=adj_features,
                        patient_encoding=patient_encoding,
                        PPI_matrix=PPI_matrix,
                        patient_features=patient_features,
                    )
                    # 计算损失
                    loss = F.mse_loss(outputs.squeeze(-1), labels)
                scaler.scale(loss).backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # 更新权重
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    afm_features=afm_features,
                    adj_features=adj_features,
                    patient_encoding=patient_encoding,
                    PPI_matrix=PPI_matrix,
                    patient_features=patient_features,
                )
                # 计算损失
                loss = F.mse_loss(outputs.squeeze(-1), labels)
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # 更新权重
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            step += 1
            train_loss += loss.item() * gradient_accumulation_steps
            train_targets.extend(labels.cpu().numpy())
            train_predictions.extend(outputs.squeeze(-1).detach().cpu().numpy())
            progress_bar.set_postfix({'loss': loss.item()})
            # 记录每个batch的损失
            wandb.log({"batch_loss": loss.item()})

        # 计算训练指标
        train_avg_loss = train_loss / len(train_loader)
        train_metrics = calculate_regression_metrics(train_targets, train_predictions)

        # 记录到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_avg_loss,
            "train_mse": train_metrics['mse'],
            "train_rmse": train_metrics['rmse'],
            "train_evs": train_metrics['evs'],
            "train_medae": train_metrics['medae'],
            "train_maxae": train_metrics['maxae'],
            "train_mape": train_metrics['mape'],
            "train_pearson": train_metrics['pearson'],
            "train_spearman": train_metrics['spearman'],
            "train_mae": train_metrics['mae'],
            "train_r2": train_metrics['r2']
        })

        # 验证
        if val_dataset is not None:
            model.eval()
            val_loss = 0
            val_targets, val_predictions = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    # 将数据移至设备
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    afm_features = batch['afm_features'].to(device)
                    adj_features = batch['adj_features'].to(device)
                    patient_encoding = batch['patient_encoding'].to(device)
                    PPI_matrix = batch['ppi_matrix'].to(device)
                    patient_features = batch['patient_features'].to(device)
                    labels = batch['labels'].to(device)

                    # 前向传播 - 验证时即使启用了混合精度也使用fp32
                    if use_mixed_precision and device != 'cpu':
                        with autocast("cuda"):
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                afm_features=afm_features,
                                adj_features=adj_features,
                                patient_encoding=patient_encoding,
                                PPI_matrix=PPI_matrix,
                                patient_features=patient_features,
                            )
                            loss = F.mse_loss(outputs.squeeze(-1), labels)
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            afm_features=afm_features,
                            adj_features=adj_features,
                            patient_encoding=patient_encoding,
                            PPI_matrix=PPI_matrix,
                            patient_features=patient_features,
                        )
                        loss = F.mse_loss(outputs.squeeze(-1), labels)
                    val_loss += loss.item()
                    val_targets.extend(labels.cpu().numpy())
                    val_predictions.extend(outputs.squeeze(-1).detach().cpu().numpy())

            val_avg_loss = val_loss / len(val_loader)
            val_metrics = calculate_regression_metrics(val_targets, val_predictions)
            # 记录到wandb
            wandb.log({
                "val_loss": val_avg_loss,
                "val_mse": val_metrics['mse'],
                "val_rmse": val_metrics['rmse'],
                "val_evs": val_metrics['evs'],
                "val_medae": val_metrics['medae'],
                "val_maxae": val_metrics['maxae'],
                "val_mape": val_metrics['mape'],
                "val_pearson": val_metrics['pearson'],
                "val_spearman": val_metrics['spearman'],
                "val_mae": val_metrics['mae'],
                "val_r2": val_metrics['r2']
            })

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}")
            # 打印结果
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Training Metrics:")
            print(f"Loss: {train_avg_loss:.4f}, MSE: {train_metrics['mse']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
            print(
                f"EVS: {train_metrics['evs']:.4f}, MedAE: {train_metrics['medae']:.4f}, MaxAE: {train_metrics['maxae']:.4f}")
            print(
                f"MAPE: {train_metrics['mape']:.4f}%, Pearson: {train_metrics['pearson']:.4f}, Spearman: {train_metrics['spearman']:.4f}")
            print(f"MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")

            print(f"\nValidation Metrics:")
            print(f"Loss: {val_avg_loss:.4f}, MSE: {val_metrics['mse']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
            print(
                f"EVS: {val_metrics['evs']:.4f}, MedAE: {val_metrics['medae']:.4f}, MaxAE: {val_metrics['maxae']:.4f}")
            print(
                f"MAPE: {val_metrics['mape']:.4f}%, Pearson: {val_metrics['pearson']:.4f}, Spearman: {val_metrics['spearman']:.4f}")
            print(f"MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")

            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                print("保存最佳模型")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Training Metrics:")
            print(f"Loss: {train_avg_loss:.4f}, MSE: {train_metrics['mse']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
            print(
                f"EVS: {train_metrics['evs']:.4f}, MedAE: {train_metrics['medae']:.4f}, MaxAE: {train_metrics['maxae']:.4f}")
            print(
                f"MAPE: {train_metrics['mape']:.4f}%, Pearson: {train_metrics['pearson']:.4f}, Spearman: {train_metrics['spearman']:.4f}")
            print(f"MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")

        # 每50个epoch保存一次检查点
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, f"checkpoint_{run_name}_epoch_{epoch + 1}.pt")
            print(f"保存检查点 epoch {epoch + 1}")

    print("训练完成!")
    return model


def predict_tcga(model, tcga_dataset, batch_size=32, device='cuda'):
    """
    对TCGA数据集进行预测

    参数:
        model: 训练好的模型
        tcga_dataset: TCGA数据集
        batch_size: 批次大小
        device: 设备

    返回:
        predictions: 预测结果列表
        original_indices: 对应的原始索引
    """
    model.eval()
    model = model.to(device)

    # 创建数据加载器
    data_collator = DataCollatorTCGA()
    tcga_loader = DataLoader(tcga_dataset, batch_size=batch_size, collate_fn=data_collator)

    predictions = []
    original_indices = []

    print("开始预测TCGA数据...")
    with torch.no_grad():
        for batch in tqdm(tcga_loader, desc="预测中"):
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            afm_features = batch['afm_features'].to(device)
            adj_features = batch['adj_features'].to(device)
            patient_encoding = batch['patient_encoding'].to(device)
            patient_features = batch['patient_features'].to(device)
            ppi_matrix = batch['ppi_matrix'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                afm_features=afm_features,
                adj_features=adj_features,
                patient_encoding=patient_encoding,
                PPI_matrix=ppi_matrix,
                patient_features=patient_features,
            )
            batch_predictions = outputs.squeeze(-1).cpu().numpy()
            predictions.extend(batch_predictions)

    return predictions


def save_predictions(predictions, tcga_response_file, output_file):
    """
    保存预测结果到CSV文件

    参数:
        predictions: 预测结果列表
        original_indices: 原始索引列表
        tcga_response_file: 原始TCGA响应文件路径
        output_file: 输出文件路径
    """
    # 读取原始TCGA文件
    tcga_df = pd.read_csv("tcga_predictions.csv")

    # 创建预测结果的DataFrame
    pred_df = pd.DataFrame({
        'predict': predictions
    })

    # pred_df = pred_df.sort_values('original_index')

    tcga_df['predict'] = predictions

    # 保存结果
    tcga_df.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")
    print(f"总共预测了 {len(predictions)} 个样本")


def get_opt(args, attr_name, default_value):
    """
    辅助函数：获取参数值
    逻辑：
    1. 如果 args 为 None，返回默认值
    2. 如果 args 中没有该属性，返回默认值
    3. 如果 args 中该属性值为 None，返回默认值
    4. 否则，返回 args 中的值
    """
    if args is None:
        return default_value
    val = getattr(args, attr_name, None)
    return val if val is not None else default_value


def TCGA(args=None):
    print("开始运行 TGCA 预测任务")
    # ================= 配置区域 =================
    # 路径配置
    model_path = get_opt(args, "model_path", "./model/multiSmiles_model/checkpoint-80000")
    tokenizer_path = get_opt(args, "tokenizer_path", "./model/custom_tokenizer")

    # 数据路径
    drug_file = get_opt(args, "drug_file", 'data/PDTC/PDTC_Drug_output.csv')
    ppi_file = get_opt(args, "sample_ppi_file", 'data/PDTC/PDTC_PPI.npy')

    patient_embed_file = get_opt(args, "sample_patient_embed_file", 'data/PDTC/PDTCSample_893_embeddings.npy')
    sensitivity_file = get_opt(args, "sample_sensitivity_file", 'data/PDTC/SampleResponse.csv')
    patient_feature_file = get_opt(args, "sample_patient_feature_file", 'data/PDTC/PDTCSample_893_laplacian.npy')
    patient_name_file = get_opt(args, "sample_patient_name_file", 'data/PDTC/PDTCSample_893_normed.csv')

    Model_patient_embed_file = get_opt(args, "model_patient_embed_file", 'data/PDTC/PDTCModel_893_embeddings.npy')
    Model_sensitivity_file = get_opt(args, "model_sensitivity_file", "data/PDTC/ModelResponse.csv")
    Model_patient_feature_file = get_opt(args, "model_patient_feature_file",
                                         'data/PDTC/PDTCModel_893_laplacian.npy')
    Model_patient_name_file = get_opt(args, "model_patient_name_file", 'data/PDTC/PDTCModel_893_normed.csv')

    # 模型与训练超参数
    num_classes = get_opt(args, "num_classes", 1)
    num_layer = get_opt(args, "num_layer", 2)
    num_heads = get_opt(args, "num_heads", 4)
    dropout_rate = get_opt(args, "dropout_rate", 0.0)
    batch_size = get_opt(args, "batch_size", 128)
    num_epochs = get_opt(args, "num_epochs", 150)
    learning_rate = get_opt(args, "learning_rate", 1e-4)
    weight_decay = get_opt(args, "weight_decay", 0.0001)
    project_name = get_opt(args, "project_name", "TCGA_predict")

    tcga_drug_file = get_opt(args, "tcga_drug_file", 'data/TCGA/TCGA_Drug.csv')
    tcga_response_file = get_opt(args, "tcga_response_file", 'data/TCGA/TCGA_response.csv')
    tcga_patient_embed_file = get_opt(args, "tcga_patient_embed_file", 'data/TCGA/TCGA893embeddings.npy')
    tcga_patient_feature_file = get_opt(args, "tcga_patient_feature_file", 'data/TCGA/TCGA_893_laplacian.npy')
    tcga_patient_name_file = get_opt(args, "tcga_patient_name_file", 'data/TCGA/TCGA_893_normed.csv')
    tcga_output_file = get_opt(args, "tcga_output_file", 'data/TCGA/TCGA_out.csv')

    run_name = project_name

    # ================= 逻辑开始 =================
    print(f"当前配置 -> Batch: {batch_size}, LR: {learning_rate}, Epochs: {num_epochs}")
    print(f"数据路径 -> {drug_file}")

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    # 创建训练数据集
    train_dataset = DrugPatientDatasetPDTC(
        drug_file=drug_file,
        patient_embed_file=patient_embed_file,
        sensitivity_file=sensitivity_file,
        ppi_file=ppi_file,
        patient_feature_file=patient_feature_file,
        patient_name_file=patient_name_file,
        tokenizer=tokenizer,
        num_classes=num_classes,
    )

    # 创建验证数据集
    val_dataset = DrugPatientDatasetPDTC(
        drug_file=drug_file,
        patient_embed_file=Model_patient_embed_file,
        sensitivity_file=Model_sensitivity_file,
        ppi_file=ppi_file,
        patient_feature_file=Model_patient_feature_file,
        patient_name_file=Model_patient_name_file,
        tokenizer=tokenizer,
        num_classes=num_classes,
    )
    full_train_dataset = ConcatDataset([train_dataset, val_dataset])

    train_size = len(full_train_dataset)
    print(f"完整训练集大小: {train_size}")

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化wandb
    wandb.init(project=project_name, name=run_name)

    # 初始化模型
    model = DrugPatientClassifier(
        drug_model_path=model_path,
        patient_encoding_dim=512,
        num_classes=num_classes,
        num_layer=num_layer,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        ParametersNum=train_dataset.patient_features_array.shape[1]
    ).to(device)

    # 训练模型
    trained_model = train_classification_model(
        model=model,
        train_dataset=full_train_dataset,
        val_dataset=None,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        weight_decay=weight_decay,
        warmup_steps=int(len(full_train_dataset) / batch_size * 5),
        project_name=project_name,
        run_name=run_name,
        use_mixed_precision=True,
        gradient_accumulation_steps=1,
        num_classes=num_classes,
    )

    print(f"模型 {run_name} 训练完成！")
    wandb.finish()

    # 创建TCGA预测数据集
    print("准备TCGA预测数据...")
    tcga_dataset = DrugPatientDatasetTCGA(
        drug_file=tcga_drug_file,
        patient_embed_file=tcga_patient_embed_file,
        sensitivity_file=tcga_response_file,
        ppi_file=ppi_file,
        patient_feature_file=tcga_patient_feature_file,
        patient_name_file=tcga_patient_name_file,
        tokenizer=tokenizer,
        num_classes=num_classes,
    )

    # 进行预测
    predictions = predict_tcga(
        model=model,
        tcga_dataset=tcga_dataset,
        batch_size=batch_size,
        device=device
    )

    # 保存预测结果
    save_predictions(
        predictions=predictions,
        tcga_response_file=tcga_response_file,
        output_file=tcga_output_file
    )

    print("所有任务完成！")


if __name__ == "__main__":
    TCGA(args=None)
