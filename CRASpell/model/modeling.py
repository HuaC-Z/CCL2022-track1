########################################
### author: liucan
### mail: kekheh_liu@163.com
### date: 2022.03.29
### 通用设置
########################################


from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from model.modeling_funcun import FuncunPreTrainedModel, prune_linear_layer
from model.modeling_funcun import FuncunAlbertConfig, FuncunBertConfig
from .lstm_crf import LSTM_CRF
from tools.cross_loss import MyCrossEntropyLoss

logger = logging.getLogger(__name__)


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}

BertLayerNorm = torch.nn.LayerNorm

class FuncunBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(FuncunBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class FuncunBertPretrained(FuncunPreTrainedModel):
    """docstring for FuncunBertPretrained"""
    base_model_prefix = "bert"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class FuncunBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(FuncunBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class FuncunBertSelfOutput(nn.Module):
    def __init__(self, config):
        super(FuncunBertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FuncunBertAttention(nn.Module):
    def __init__(self, config):
        super(FuncunBertAttention, self).__init__()
        self.self = FuncunBertSelfAttention(config)
        self.output = FuncunBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class FuncunBertIntermediate(nn.Module):
    def __init__(self, config):
        super(FuncunBertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class FuncunBertOutput(nn.Module):
    def __init__(self, config):
        super(FuncunBertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FuncunBertLayer(nn.Module):
    def __init__(self, config):
        super(FuncunBertLayer, self).__init__()
        self.attention = FuncunBertAttention(config)
        self.intermediate = FuncunBertIntermediate(config)
        self.output = FuncunBertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs
        

class FuncunBertEncoder(nn.Module):
    def __init__(self, config):
        super(FuncunBertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([FuncunBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class FuncunBertPooler(nn.Module):
    def __init__(self, config):
        super(FuncunBertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class FuncunBertModel(FuncunBertPretrained):
    
    def __init__(self, config):
        super(FuncunBertModel, self).__init__(config)

        self.embeddings = FuncunBertEmbeddings(config)
        self.encoder = FuncunBertEncoder(config)
        self.pooler = FuncunBertPooler(config)

        self.init_weights()


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
        

class FuncunCRASpell(FuncunBertPretrained):
    """docstring for FuncunCRASpell"""
    def __init__(self, config):
        super(FuncunCRASpell, self).__init__(config)
        self.bert = FuncunBertModel(config)
        self.gen_linear = nn.Linear(config.hidden_size, config.vocab_size)

        self.copy_linear1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.copy_linear2 = nn.Linear(config.hidden_size // 2, 1)

        self.LayerNorm1 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size // 2, eps=config.layer_norm_eps)
        self.activation = ACT2FN[config.hidden_act]

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cross_loss = MyCrossEntropyLoss()
        self.vocab_size = config.vocab_size
        self.alpha = config.alpha

    def forward(self, input_ids, noise_ids, attention_mask, label_ids,
        token_type_ids=None, position_ids=None, head_mask=None):

        logger.debug("input_ids: {}, label_ids: {}, noise_ids: {}".format(
            input_ids, label_ids, noise_ids))

        last_input_hidden_state = self.bert(input_ids = input_ids,
            attention_mask = attention_mask, token_type_ids = token_type_ids,
            position_ids = position_ids, head_mask = head_mask)[0]
        
        logger.debug("aa: {}\nattention_mask: {}".format(
            torch.equal(input_ids, noise_ids), attention_mask))
        nosie_mask = (input_ids == noise_ids) * attention_mask

        input_hidden_state = self.gen_linear(self.dropout(last_input_hidden_state))

        p_g = F.softmax(input_hidden_state, dim = 2)

        # batch_size, seq_len, vocab_size
        input_p_g = F.log_softmax(input_hidden_state, dim = 2)

        logger.debug("\ninput_p_g: {}".format(input_p_g))

        hidden_state = self.LayerNorm1(last_input_hidden_state)
        hidden_state = self.copy_linear1(self.dropout(hidden_state))
        hidden_state = self.LayerNorm2(self.activation(hidden_state))
        
        # batch_size, seq_len, 1
        omega = torch.sigmoid(self.copy_linear2(self.dropout(hidden_state)))
        omega = torch.matmul(omega, 
            torch.ones([1, self.vocab_size], dtype = omega.dtype).to(omega.device))

        logger.debug("\nomega: {}".format(omega))

        last_noise_hidden_state = self.bert(input_ids = noise_ids,
            attention_mask = attention_mask, token_type_ids = token_type_ids,
            position_ids = position_ids, head_mask = head_mask)[0]

        nosie_p_g = F.log_softmax(self.gen_linear(self.dropout(last_noise_hidden_state)),
            dim = 2)

        kl_loss = self._kl_loss(nosie_p_g, input_p_g, nosie_mask)

        # label loss
        p_c = F.one_hot(input_ids, num_classes = self.vocab_size)

        output_p = omega * p_c + (1.0 - omega) * p_g
        output_p = torch.clamp(output_p, 1e-10, 1.0 - 1e-7)

        logger.debug("\noutput_p size: {}, label_ids size: {}".format(output_p.size(), label_ids.size()))
        cross_loss = self.cross_loss(output_p, label_ids, attention_mask)

        logger.debug("cross_loss: {}, kl_loss: {}, alpha: {}".format(
            cross_loss, kl_loss, self.alpha))
        loss = self.alpha * kl_loss + (1 - self.alpha) * cross_loss

        output = output_p.argmax(dim = 2)

        return (loss, output, kl_loss, cross_loss)

    def predict(self, input_ids, attention_mask, 
        token_type_ids=None, position_ids=None, head_mask=None):

        last_input_hidden_state = self.bert(input_ids = input_ids,
            attention_mask = attention_mask, token_type_ids = token_type_ids,
            position_ids = position_ids, head_mask = head_mask)[0]

        p_g = F.softmax(self.gen_linear(last_input_hidden_state), dim = 2)

        hidden_state = self.LayerNorm1(last_input_hidden_state)
        hidden_state = self.copy_linear1(hidden_state)
        hidden_state = self.LayerNorm2(self.activation(hidden_state))
        
        # batch_size, seq_len, 1
        omega = torch.sigmoid(self.copy_linear2(hidden_state))
        omega = torch.matmul(omega, 
            torch.ones([1, self.vocab_size], dtype = omega.dtype).to(omega.device))

        p_c = F.one_hot(input_ids, num_classes = self.vocab_size)
        output_p = omega * p_c + (1.0 - omega) * p_g

        output = output_p.argmax(dim = 2)

        return output


    def _kl_loss(self, prob1, prob2, nosie_mask):

        p1 = torch.exp(prob1)
        p2 = torch.exp(prob2)

        neg_ent_1 = (p1 * prob1).sum(dim=-1)
        neg_cross_ent_1 = (p1 * prob2).sum(dim=-1)

        neg_ent_2 = (p2 * prob2).sum(dim=-1)
        neg_cross_ent_2  = (p2 * prob1).sum(dim=-1)

        kl_1 = (neg_ent_1 - neg_cross_ent_1)
        kl_2 = (neg_ent_2 - neg_cross_ent_2)

        logger.debug("\nkl_1: {}, kl_2: {}, kl_1.size: {}".format(kl_1, kl_2, 
            kl_1.size()))

        kl_loss = (kl_1 + kl_2) / 2.0
        logger.debug("\nkl_loss: {}, nosie_mask: {}".format(kl_loss, nosie_mask))
        loss = (kl_loss * nosie_mask).sum() / nosie_mask.sum()

        return loss 






    




        