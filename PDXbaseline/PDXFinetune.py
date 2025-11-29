import json
import os
import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, get_constant_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, auc, \
    precision_recall_curve
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from torch.amp import autocast
from sklearn.model_selection import KFold
import pandas as pd

from PDXbaseline.DrugPatientClassifier import DrugPatientClassifier
from PDXbaseline.DrugPatientDatasetPDX import DrugPatientDatasetPDX, DataCollatorPDX

os.environ["WANDB_MODE"] = "offline"

import warnings

warnings.simplefilter("ignore", category=FutureWarning)


def train_fold(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, use_mixed_precision,
               gradient_accumulation_steps, num_classes, fold_idx, run_nam,regression_model=None,):
    """
    训练特定折的模型
    """
    best_val_f1 = 0
    best_val_auc = 0
    early_stopping_patience = 1000
    early_stopping_counter = 0
    fold_results = {}

    # 设置混合精度训练
    if use_mixed_precision and device != 'cpu':
        from torch.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_probs = []
        train_labels = []

        # 训练循环
        progress_bar = tqdm(train_loader, desc=f"Fold {fold_idx + 1} Epoch {epoch + 1}/{num_epochs}")
        step = 0

        for batch in progress_bar:
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            afm_features = batch['afm_features'].to(device)
            adj_features = batch['adj_features'].to(device)
            patient_encoding = batch['patient_encoding'].to(device)
            match regression_model:
                case None:
                    patient_features = batch['patient_features'].to(device)
                    PPI_matrix = batch['ppi_matrix'].to(device)
                case 'HyperAT':
                    patient_features = batch['patient_features'].to(device)
                    PPI_matrix = None
                case 'PPIAT':
                    PPI_matrix = batch['ppi_matrix'].to(device)
                    patient_features = None
                case 'GeneAT':
                    PPI_matrix = None
                    patient_features = None
                case _:
                    patient_features = batch['patient_features'].to(device)
                    PPI_matrix = batch['ppi_matrix'].to(device)
            labels = batch['labels'].to(device)

            # 清除梯度
            optimizer.zero_grad()

            # 使用混合精度训练
            if use_mixed_precision and device != 'cpu':
                with autocast("cuda"):
                    # 前向传播
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        afm_features=afm_features,
                        adj_features=adj_features,
                        patient_encoding=patient_encoding,
                        PPI_matrix=PPI_matrix,
                        patient_features=patient_features,
                    )
                    # 计算损失
                    if num_classes == 2:
                        # 二分类
                        logits = logits.squeeze(-1)
                        labels = labels.float()
                        loss = F.binary_cross_entropy_with_logits(logits, labels)
                    else:
                        loss = F.cross_entropy(logits, labels)

                # 使用scaler进行反向传播和参数更新
                scaler.scale(loss).backward()

                # 只有在完成了梯度累积步数后才更新模型参数
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # 更新权重
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()  # 清除梯度
                    scheduler.step()  # 更新学习率
            else:
                # 标准的前向传播和反向传播
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    afm_features=afm_features,
                    adj_features=adj_features,
                    patient_encoding=patient_encoding,
                    PPI_matrix=PPI_matrix,
                    patient_features=patient_features,
                )
                # 计算损失
                if num_classes == 2:
                    # 二分类
                    logits = logits.squeeze(-1)
                    labels = labels.float()
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                else:
                    loss = F.cross_entropy(logits, labels)
                # 反向传播
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # 更新权重
                    optimizer.step()
                    optimizer.zero_grad()  # 清除梯度
                    scheduler.step()  # 更新学习率

            step += 1

            train_loss += loss.item() * gradient_accumulation_steps  # 调整回实际损失值

            # 收集预测结果
            if num_classes == 2:
                probs = torch.sigmoid(logits).detach().cpu().numpy()  # shape: [batch_size]
                preds = (probs >= 0.5).astype(int)  # 阈值0.5划分为0或1
                train_probs.extend(probs)
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())
            else:
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()  # 获取概率预测值
                preds = np.argmax(probs, axis=1)
                train_probs.extend(probs)
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())

            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})

            # 记录每个batch的损失
            wandb.log({"batch_loss": loss.item()})

        # 计算训练指标
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_precision = precision_score(train_labels, train_preds, average='macro')
        train_recall = recall_score(train_labels, train_preds, average='macro')
        train_avg_loss = train_loss / len(train_loader)

        if num_classes == 2:
            train_auc = roc_auc_score(train_labels, train_probs)
            precision_curve, recall_curve, _ = precision_recall_curve(train_labels, train_probs)
            train_aupr = auc(recall_curve, precision_curve)
        else:
            # 计算训练AUC（多分类采用OVR策略）
            train_auc = roc_auc_score(
                np.eye(model.classifier[-1].out_features)[train_labels],  # one-hot编码标签
                np.array(train_probs),
                average='macro',
                multi_class='ovr'
            )
            # one-hot 编码标签
            train_labels_bin = label_binarize(train_labels, classes=range(num_classes))
            # === Macro AUPR ===
            aupr_list = []
            for i in range(num_classes):
                precision, recall, _ = precision_recall_curve(train_labels_bin[:, i], np.array(train_probs)[:, i])
                aupr = auc(recall, precision)
                aupr_list.append(aupr)
            train_aupr = np.mean(aupr_list)

            # === Micro AUPR ===
            precision_micro, recall_micro, _ = precision_recall_curve(
                train_labels_bin.ravel(), np.array(train_probs).ravel()
            )
            train_aupr_micro = auc(recall_micro, precision_micro)

        # 验证
        val_metrics = evaluate_model(model, val_loader, device, num_classes, use_mixed_precision)
        val_avg_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1']
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        val_auc = val_metrics['auc']
        val_aupr = val_metrics['aupr']

        # 记录到wandb
        wandb.log({
            "fold": fold_idx + 1,
            "epoch": epoch + 1,
            "train_loss": train_avg_loss,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_auc": train_auc,
            "train_aupr": train_aupr,
            "val_loss": val_avg_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_auc": val_auc,
            "val_aupr": val_aupr,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        # 打印结果
        print(f"Fold {fold_idx + 1} - Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_avg_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val AUC: {val_auc:.4f}")

    # 打印最后一轮结果
    print(f"Fold {fold_idx + 1} 训练完成，共 {epoch + 1} 轮")
    print(f"最后一轮结果: Val Loss: {val_avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    val_metrics['last_epoch'] = epoch + 1
    return val_metrics


def evaluate_model(model, dataloader, device, num_classes, use_mixed_precision=True,
                   regression_model=None):
    """
    评估模型，返回指标字典
    """
    model.eval()
    val_loss = 0
    val_preds = []
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            afm_features = batch['afm_features'].to(device)
            adj_features = batch['adj_features'].to(device)
            patient_encoding = batch['patient_encoding'].to(device)
            match regression_model:
                case None:
                    patient_features = batch['patient_features'].to(device)
                    PPI_matrix = batch['ppi_matrix'].to(device)
                case 'HyperAT':
                    patient_features = batch['patient_features'].to(device)
                    PPI_matrix = None
                case 'PPIAT':
                    PPI_matrix = batch['ppi_matrix'].to(device)
                    patient_features = None
                case 'GeneAT':
                    PPI_matrix = None
                    patient_features = None
                case _:
                    patient_features = batch['patient_features'].to(device)
                    PPI_matrix = batch['ppi_matrix'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播 - 验证时即使启用了混合精度也使用fp32
            if use_mixed_precision and device != 'cpu':
                with autocast("cuda"):
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        afm_features=afm_features,
                        adj_features=adj_features,
                        patient_encoding=patient_encoding,
                        PPI_matrix=PPI_matrix,
                        patient_features=patient_features,
                    )
                    if num_classes == 2:
                        # 二分类
                        logits = logits.squeeze(-1)
                        labels = labels.float()
                        # print(logits)
                        loss = F.binary_cross_entropy_with_logits(logits, labels)
                    else:
                        # 多分类
                        loss = F.cross_entropy(logits, labels)
            else:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    afm_features=afm_features,
                    adj_features=adj_features,
                    patient_encoding=patient_encoding,
                    PPI_matrix=PPI_matrix,
                    patient_features=patient_features,
                )
                if num_classes == 2:
                    # 二分类
                    logits = logits.squeeze(-1)
                    labels = labels.float()
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                else:
                    # 多分类
                    loss = F.cross_entropy(logits, labels)

            val_loss += loss.item()

            if num_classes == 2:
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
            else:
                logits_softmax = F.softmax(logits, dim=1).detach().cpu().numpy()
                preds = np.argmax(logits_softmax, axis=1)
                val_probs.extend(logits_softmax)
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

    # 计算验证指标
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_precision = precision_score(val_labels, val_preds, average='macro')
    val_recall = recall_score(val_labels, val_preds, average='macro')
    val_avg_loss = val_loss / len(dataloader)

    if num_classes == 2:
        val_auc = roc_auc_score(val_labels, val_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(val_labels, val_probs)
        val_aupr = auc(recall_curve, precision_curve)
    else:
        # 计算验证AUC（多分类采用OVR策略）
        val_auc = roc_auc_score(
            np.eye(num_classes)[val_labels],  # one-hot编码标签
            np.array(val_probs),
            average='macro',
            multi_class='ovr'
        )
        # one-hot 编码标签
        val_labels_bin = label_binarize(val_labels, classes=range(num_classes))
        # === Macro AUPR ===
        aupr_list = []
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(val_labels_bin[:, i], np.array(val_probs)[:, i])
            aupr = auc(recall, precision)
            aupr_list.append(aupr)
        val_aupr = np.mean(aupr_list)

    metrics = {
        'loss': val_avg_loss,
        'accuracy': val_acc,
        'f1': val_f1,
        'precision': val_precision,
        'recall': val_recall,
        'auc': val_auc,
        'aupr': val_aupr
    }

    return metrics


def train_cross_validation(model_init_func, dataset, batch_size, num_epochs, learning_rate,
                           device, weight_decay, project_name, run_name, use_mixed_precision=True,
                           gradient_accumulation_steps=1, num_classes=4, regression_model=None
                           ,n_folds=10):
    """
    使用k-fold交叉验证训练模型

    参数:
        model_init_func: 用于初始化模型的函数
        dataset: 完整数据集
        batch_size: 批次大小
        num_epochs: 每折训练的轮数
        learning_rate: 学习率
        device: 训练设备
        weight_decay: 权重衰减
        project_name: wandb项目名称
        run_name: wandb运行名称
        use_mixed_precision: 是否使用混合精度训练
        gradient_accumulation_steps: 梯度累积步数
        num_classes: 类别数量
        n_folds: 交叉验证折数
    """
    # 初始化用于存储所有折结果的列表
    all_fold_results = []

    # 创建K-fold交叉验证拆分器
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 初始化wandb
    wandb.init(project=project_name, name=f"{run_name}_cv", reinit=True)

    # 配置wandb
    wandb.config.update({
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "n_folds": n_folds,
        "mixed_precision": use_mixed_precision
    })

    # 准备数据加载器
    data_collator = DataCollatorPDX()

    # 循环进行k折交叉验证
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        print(f"\n======== Fold {fold_idx + 1}/{n_folds} ========")

        # 创建当前折的数据加载器
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=True
        )

        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=data_collator
        )

        model = model_init_func().to(device)

        # 分层学习率 - 为不同组件设置不同的学习率
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

        optimizer = AdamW(optimizer_grouped_parameters)

        # 更新学习率调度器的总步数，考虑梯度累积
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1)
        )
        # 训练当前折
        fold_results = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            use_mixed_precision=use_mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_classes=num_classes,
            regression_model=regression_model,
            fold_idx=fold_idx,
            run_name=run_name
        )

        # 记录每一折的最后一轮结果
        all_fold_results.append(fold_results)

        # 记录到wandb
        for metric_name, metric_value in fold_results.items():
            wandb.log({f"fold_{fold_idx + 1}_{metric_name}": metric_value})

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # 计算并展示所有折的平均结果
    avg_results = {}
    for metric in all_fold_results[0].keys():
        avg_results[f"avg_{metric}"] = np.mean([fold[metric] for fold in all_fold_results])
        avg_results[f"std_{metric}"] = np.std([fold[metric] for fold in all_fold_results])

    # 记录到wandb
    wandb.log(avg_results)

    # 创建结果表格并保存每一折的最终结果
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(f"{run_name}.csv", index=False)

    # 打印平均结果
    print("\n===== 交叉验证平均结果 =====")
    for metric, value in avg_results.items():
        print(f"{metric}: {value:.4f}")
    wandb.finish()

    return avg_results, all_fold_results


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


def PDXFinetune(args=None):
    print("开始运行 PDTCRegressionTenfold 任务")
    # ================= 配置区域 =================
    # 路径配置
    model_path = get_opt(args, "model_path", "./model/multiSmiles_model/checkpoint-80000")
    tokenizer_path = get_opt(args, "tokenizer_path", "./model/custom_tokenizer")

    # 数据路径
    drug_file = get_opt(args, "drug_file", 'data/PDX/PDX_drugOutput.csv')
    ppi_file = get_opt(args, "sample_ppi_file", 'data/PDX/933standard_adjacency_score_matrix.npy')
    patient_embed_file = get_opt(args, "sample_patient_embed_file", 'data/PDX/PDX933_embeddings.npy')
    sensitivity_file = get_opt(args, "sample_sensitivity_file", 'data/PDX/PDX_sensitivity.csv')
    patient_feature_file = get_opt(args, "sample_patient_feature_file", 'data/PDX/PDX_933_laplacian.npy')
    patient_name_file = get_opt(args, "sample_patient_name_file", 'data/PDX/PDX_embed.csv')
    model_file = get_opt(args, "model_file", 'data/PDX/PDX_model.csv')

    # 模型与训练超参数
    num_classes = get_opt(args, "num_classes", 2)
    num_layer = get_opt(args, "num_layer", 2)
    num_heads = get_opt(args, "num_heads", 4)
    dropout_rate = get_opt(args, "dropout_rate", 0.0)
    batch_size = get_opt(args, "batch_size", 128)
    num_epochs = get_opt(args, "num_epochs", 50)
    learning_rate = get_opt(args, "learning_rate", 1e-4)
    weight_decay = get_opt(args, "weight_decay", 0.0001)
    project_name = get_opt(args, "project_name", "PDXFinetuneTen")

    regression_model = get_opt(args, "regression_model", None)


    # ================= 逻辑开始 =================

    print(f"当前配置 -> Batch: {batch_size}, LR: {learning_rate}, Epochs: {num_epochs}")
    print(f"数据路径 -> {drug_file}")

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    # 创建完整数据集
    full_dataset = DrugPatientDatasetPDX(
        drug_file=drug_file,
        patient_embed_file=patient_embed_file,
        sensitivity_file=sensitivity_file,
        model_file=model_file,
        ppi_file=ppi_file,
        patient_feature_file=patient_feature_file,
        patient_name_file=patient_name_file,
        tokenizer=tokenizer,
        num_classes=num_classes,
    )

    print(f"完整数据集大小: {len(full_dataset)}")

    # 训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型初始化函数
    def init_model():
        return DrugPatientClassifier(
            drug_model_path=model_path,
            patient_encoding_dim=512,
            num_classes=num_classes,
            num_layer=num_layer,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            ParametersNum=933,
        )

    # 执行交叉验证
    avg_results, all_fold_results = train_cross_validation(
        model_init_func=init_model,
        dataset=full_dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        weight_decay=weight_decay,
        project_name=project_name,
        run_name=project_name,
        regression_model=regression_model,
        use_mixed_precision=True,
        gradient_accumulation_steps=1,
        num_classes=num_classes,
        n_folds=10
    )

    print("交叉验证完成!")

    # 保存交叉验证结果到JSON
    results_dict = {
        'avg_results': {k: float(v) for k, v in avg_results.items()},
        'fold_results': all_fold_results
    }

    with open(f"{project_name}.json", "w") as f:
        json.dump(results_dict, f, indent=2)


if __name__ == "__main__":
    PDXFinetune(args=None)
