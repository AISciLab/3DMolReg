import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.amp import autocast 
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_atom_importance_from_classifier(model, batch):

    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            afm_features=batch['afm_features'],
            adj_features=batch['adj_features'],
            patient_encoding=batch['patient_encoding'],
            PPI_matrix=batch['ppi_matrix'],
            patient_features=batch['patient_features'],
            output_drug_attentions=True  
        )
        last_layer_attentions = outputs.drug_attentions[-1]

        cls_token_attentions = last_layer_attentions[:, :, 0, :]

        avg_cls_attentions = cls_token_attentions.mean(dim=1)

        text_length = outputs.drug_text_length
        atom_attentions_unmasked = avg_cls_attentions

        knowledge_mask = torch.any(batch['afm_features'] != 0, dim=-1).int()

        patient_pairwise_attention = outputs.patient_pairwise_attention

        patient_sequence_importance = outputs.patient_sequence_importance

        batch_importance_scores = []
        for i in range(atom_attentions_unmasked.shape[0]):
            sample_scores = atom_attentions_unmasked[i]
            real_atom_scores = sample_scores
            batch_importance_scores.append(real_atom_scores)
    return batch_importance_scores , patient_pairwise_attention, patient_sequence_importance
