import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaConfig
from transformers.modeling_outputs import MaskedLMOutput

import warnings
warnings.simplefilter("ignore", category=FutureWarning)


class MultiSmilesModel(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)

        self.afm_projection = nn.Linear(config.afm_dim, config.hidden_size)
        self.adj_projection = nn.Linear(config.adj_dim, config.hidden_size)

        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            afm_features=None,  # [batch_size, n, afm_dim]
            adj_features=None,  # [batch_size, n, adj_dim]
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        text_embeds = self.roberta.embeddings(
            input_ids=None,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )  # [batch_size, seq_len, hidden_size]

        text_length = text_embeds.size(1)

        if afm_features is not None and adj_features is not None:
            afm_projected = self.afm_projection(afm_features)  # [batch_size, n, hidden_size]
            adj_projected = self.adj_projection(adj_features)  # [batch_size, n, hidden_size]

            knowledge_embeds = afm_projected + adj_projected  # [batch_size, n, hidden_size]
            combined_embeds = torch.cat([text_embeds, knowledge_embeds], dim=1)  # [batch_size, seq_len+n, hidden_size]

            # 为额外的知识token创建扩展的注意力掩码
            if attention_mask is not None:
                knowledge_attention = torch.ones(
                    (attention_mask.shape[0], afm_features.shape[1]),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                extended_attention_mask = torch.cat([attention_mask, knowledge_attention], dim=1)
            else:
                extended_attention_mask = None
        else:
            combined_embeds = text_embeds
            extended_attention_mask = attention_mask

        if extended_attention_mask is not None:
            extended_attention_mask = self.roberta.get_extended_attention_mask(
                extended_attention_mask,
                combined_embeds.size()[:2],
                combined_embeds.device
            )

        encoder_outputs = self.roberta.encoder(
            hidden_states=combined_embeds,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state

        outputs = (sequence_output, text_length)

        if not return_dict:
            return outputs
            
        from collections import namedtuple
        MultimodalOutput = namedtuple(
            "MultimodalOutput",
            ["last_hidden_state", "text_length","hidden_states", "attentions"]
        )

        return MultimodalOutput(
            last_hidden_state=sequence_output,
            text_length=text_length,
            hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, "hidden_states") else None,
            attentions=encoder_outputs.attentions if hasattr(encoder_outputs, "attentions") else None,
        )


class MultiSmilesModelForMaskedLM(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = MultiSmilesModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            afm_features=None,
            adj_features=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            afm_features=afm_features,
            adj_features=adj_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        text_length = outputs.text_length

        text_sequence_output = sequence_output[:, :text_length, :]

        prediction_scores = self.lm_head(text_sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_pretrained(self, save_directory, safe_serialization=False, **kwargs):
        kwargs["safe_serialization"] = False
        super().save_pretrained(save_directory, **kwargs)

class MultiSmilesModelConfig(RobertaConfig):
    model_type = "multimodel-smiles"

    def __init__(self, afm_dim=27, adj_dim=3, **kwargs):
        super().__init__(**kwargs)
        self.afm_dim = afm_dim

        self.adj_dim = adj_dim
