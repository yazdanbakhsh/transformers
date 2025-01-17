# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# AMIR: TEST
"""PyTorch OpenAI GPT-2 model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

# amir
from .soft_thres_layer import soft_thres_layer

EARLY_STOP_FLAG = False
QUANT_FLAG = False
PRUN_FLAG = True
KBIT = 8
PCT = 1 / 2
REUSE_BY_UNUSED_SLOT_FLAG = True
USE_ISCA_PRUN = False
USE_RRAM = True
# rima

if version.parse(torch.__version__) >= version.parse("1.6"):
  is_amp_available = True
  from torch.cuda.amp import autocast
else:
  is_amp_available = False

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from ...utils import logging
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gpt2 import GPT2Config

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
  """Load tf checkpoints in a pytorch model"""
  try:
    import re

    import tensorflow as tf
  except ImportError:
    logger.error(
        "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
        "https://www.tensorflow.org/install/ for installation instructions.")
    raise
  tf_path = os.path.abspath(gpt2_checkpoint_path)
  logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
  # Load weights from TF model
  init_vars = tf.train.list_variables(tf_path)
  names = []
  arrays = []
  for name, shape in init_vars:
    logger.info(f"Loading TF weight {name} with shape {shape}")
    array = tf.train.load_variable(tf_path, name)
    names.append(name)
    arrays.append(array.squeeze())

  for name, array in zip(names, arrays):
    name = name[6:]  # skip "model/"
    name = name.split("/")
    pointer = model
    for m_name in name:
      if re.fullmatch(r"[A-Za-z]+\d+", m_name):
        scope_names = re.split(r"(\d+)", m_name)
      else:
        scope_names = [m_name]
      if scope_names[0] == "w" or scope_names[0] == "g":
        pointer = getattr(pointer, "weight")
      elif scope_names[0] == "b":
        pointer = getattr(pointer, "bias")
      elif scope_names[0] == "wpe" or scope_names[0] == "wte":
        pointer = getattr(pointer, scope_names[0])
        pointer = getattr(pointer, "weight")
      else:
        pointer = getattr(pointer, scope_names[0])
      if len(scope_names) >= 2:
        num = int(scope_names[1])
        pointer = pointer[num]
    try:
      assert (
          pointer.shape == array.shape
      ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
    except AssertionError as e:
      e.args += (pointer.shape, array.shape)
      raise
    logger.info(f"Initialize PyTorch weight {name}")
    pointer.data = torch.from_numpy(array)
  return model


class GPT2Attention(nn.Module):

  def __init__(self, config, is_cross_attention=False, layer_idx=None):
    super().__init__()
    max_positions = config.max_position_embeddings
    # `register_buffer`: This is typically used to register a buffer that
    # should not to be considered a model parameter.
    self.register_buffer(
        "bias",
        torch.tril(
            torch.ones((max_positions, max_positions),
                       dtype=torch.uint8)).view(1, 1, max_positions,
                                                max_positions),
    )
    self.register_buffer("masked_bias", torch.tensor(-1e4))

    self.embed_dim = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.embed_dim // self.num_heads
    self.split_size = self.embed_dim
    if self.head_dim * self.num_heads != self.embed_dim:
      raise ValueError(
          f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
      )

    self.scale_attn_weights = config.scale_attn_weights
    self.is_cross_attention = is_cross_attention

    # Layer-wise attention scaling, reordering, and upcasting
    self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
    self.layer_idx = layer_idx
    self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

    if self.is_cross_attention:
      self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
      self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
    else:
      self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
    self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

    self.attn_dropout = nn.Dropout(config.attn_pdrop)
    self.resid_dropout = nn.Dropout(config.resid_pdrop)

    self.pruned_heads = set()

    # amir
    self.prun = PRUN_FLAG
    if self.prun:
      # ALPHA: we need to tune alpha.
      # Change C to the same value as casual mask.
      self.soft_thres_layer = soft_thres_layer(
          s=10.0, c=-1e4, alpha=config.threshold)
    self.early_stop = config.early_stopping
    self.quant = QUANT_FLAG
    self.kbit = KBIT
    self.six_sigma = dict()
    # rima

  # amir
  def quantize_by_bit(self, w, bit_num, alpha_key, ori_bit_num):
    alpha = self.six_sigma.get(alpha_key)
    if bit_num == 1:
      return torch.zeros_like(w).cuda()

    w = torch.div(w, alpha)
    w = w.clamp(min=-1, max=1)
    factor = (w * (2**(ori_bit_num - 1) - 1)).round()
    max_factor = 2**((ori_bit_num - 1) - (bit_num - 1))
    max_value = (
        (factor // max_factor) * max_factor) / (2**(ori_bit_num - 1) - 1)
    return max_value * alpha

  def quantize(self, w, bit_num=12, alpha_key=None):
    """Computes the quantized value of tensor w.

    Args:
      w: input argument.
      bit_num: Number of bits, at least 2.
      alpha_key: key value representing the target vector.

    Returns:
      Quantized values.
    """
    if self.six_sigma.get(alpha_key) is None:
      std, _ = torch.std_mean(w)

      if alpha_key == "k" or alpha_key == "q":
        alpha = 6.0 * std
        print(f"Number of bit for {alpha_key}: ", bit_num)
    #   if alpha_key == "scores":
    #     alpha = 4 * std
    #     bit_num = 24
    #   if alpha_key == "scores_softmax":
    #     alpha = 1.0
    #     bit_num = 16
    #   if alpha_key == "v":
    #     alpha = 13 * std
    #     bit_num = 16
    #   if alpha_key == "out":
    #     alpha = 15 * std
    #     bit_num = 20
      if alpha_key == 'scores':
        alpha = 4 * std
        # bit_num = 24
        # bit_num = 12
        print('number of bit for scores', bit_num)
      if alpha_key == 'scores_softmax':
        alpha = 1.0
        # bit_num = 16
        # bit_num = 8
        print('number of bit for softmax', bit_num)
      if alpha_key == 'v':
        alpha = 13 * std
        # bit_num = 16
        # bit_num = 8
        print('number of bit for v', bit_num)
      if alpha_key == 'out':
        alpha = 15 * std
        # bit_num = 20
        # bit_num = 16
        print('number of bit for out', bit_num)
      self.six_sigma[alpha_key] = alpha
      print("dict key:", alpha_key, "added alpha", alpha)
    else:
      alpha = self.six_sigma.get(alpha_key)
    # print("w is: ", w)
    w = torch.div(w, alpha)
    w = w.clamp(min=-1, max=1)
    # Quantization, (bit_num - 1) for the sign bit.
    w = (w * (2**(bit_num - 1) - 1)).round() / (2**(bit_num - 1) - 1)
    return w * alpha
  # rima

  def prune_heads(self, heads):
    if len(heads) == 0:
      return
    heads, index = find_pruneable_heads_and_indices(heads, self.num_heads,
                                                    self.head_dim,
                                                    self.pruned_heads)
    index_attn = torch.cat(
        [index, index + self.split_size, index + (2 * self.split_size)])

    # Prune conv1d layers
    self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
    self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

    # Update hyper params
    self.split_size = (self.split_size // self.num_heads) * (
        self.num_heads - len(heads))
    self.num_heads = self.num_heads - len(heads)
    self.pruned_heads = self.pruned_heads.union(heads)

  def _attn(self, query, key, value, attention_mask=None, head_mask=None):
    # GPT2-Large
    # Num Heads = 20
    # transpose(-1, -2): [Batch, 20, 1024, 64] -> [Batch, 20, 64, 1024]
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    # attn_weights: [Batch, 20, 1024, 1024]

    # amir: only scale if we don't do pruning.
    if (not self.prun) and (not self.quant) and (not self.early_stop):
      #assert False, "Oh no! we are not running this!"
      if self.scale_attn_weights:
        attn_weights = attn_weights / (value.size(-1)**0.5)
    # rima
    # Layer-wise attention scaling (amir: default FALSE)
      if self.scale_attn_by_inverse_layer_idx:
        assert False, "ISCA: Oh No! We haven't done this!"
        attn_weights = attn_weights / float(self.layer_idx + 1)

      if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        # AMIR-ISCA: This is running! Not cross attention!
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length -
                                query_length:key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights,
                                   self.masked_bias.to(attn_weights.dtype))
    if not self.is_cross_attention:
      # if only "normal" attention layer implements causal mask
      # AMIR-ISCA: This is running! Not cross attention!
      query_length, key_length = query.size(-2), key.size(-2)
      causal_mask = self.bias[:, :, key_length -
                              query_length:key_length, :key_length].bool()
      attn_weights = torch.where(causal_mask, attn_weights,
                                 self.masked_bias.to(attn_weights.dtype))
    if attention_mask is not None:
      assert torch.count_nonzero(attention_mask) == 0, "All are zero!"
      # amir
      attention_mask[attention_mask == -10000] = -10000
      # rima
      # Apply the attention mask: [Batch, 1, 1, 1024]
      attn_weights = attn_weights + attention_mask
      # Zheng added
      # attn_weights = attn_weights + attention_mask.transpose(2,3)

    # print("attention max: ", torch.max(attn_weights))
    # print("attention min: ", torch.min(attn_weights))
    # Zheng MASK [B, 1, 1, 1024]
    # amir
    var = 0
    sigmoid = nn.Sigmoid()
    new_attention_weights = None
    # Amir: Using the actual mask
    query_length, key_length = query.size(-2), key.size(-2)
    my_causal_mask = self.bias[:, :, key_length -
                               query_length:key_length, :key_length].bool()
    # Actual Mask: [1, 1, 1024, 1024]
    my_actual_mask = torch.where(my_causal_mask,
                                 torch.tensor(0.).to(attn_weights.dtype).cuda(),
                                 self.masked_bias.to(attn_weights.dtype))
    # print("Actual Mask Size: ", my_actual_mask.size())
    # my_actual_mask [1, 1, 1024, 1024]
    # Rim

    if self.quant:
      mykey = "q"
      newq = self.quantize(query, bit_num=self.kbit, alpha_key=mykey)
      if self.early_stop:
        new_attention_weights = torch.zeros(
            (newq.shape[0], newq.shape[1], newq.shape[2],
             newq.shape[2])).cuda()
      else:
        # assert False, "Not happening for now!"
        mykey = "k"
        newkey = self.quantize(key, bit_num=self.kbit, alpha_key=mykey)
        attn_weights = torch.matmul(newq, newkey.transpose(-1, -2)) / (
            value.size(-1)**0.5)
      if attention_mask is not None:
        assert torch.count_nonzero(attention_mask) == 0, "All are zero!"
        attention_mask[attention_mask == -10000] = -10000
        attn_weights = attn_weights + attention_mask
        if new_attention_weights is not None:
          new_attention_weights = new_attention_weights + attention_mask
          # Zheng added ----------------------------------------------------------
          # new_attention_weights = new_attention_weights + attention_mask.transpose(2,3)
        # new_attention_weights = self.quantize(new_attention_weights,)
    if self.prun and self.early_stop:
      self.sparsity = [0 for _ in range(self.kbit)]
      same_sign = torch.sign(key) * torch.sign(query)
      same_sign = torch.where(same_sign > 0, 1, 0)
      q_abs_sum = torch.sum(torch.abs(query) * same_sign, 3)
      # query size: [8, 20, 1024, 64]
      # print("ISCAREMOVE: query Size: ", query.size())
      # q_abs_sum size: [8, 20, 1024]
      # print("ISCAREMOVE: q_abs_sum Size: ", q_abs_sum.size())
      bound_sum = 0
      bound_bit = torch.zeros(self.kbit)
      # Following lines pre-calculate the remaining maximum K value
      # after completeing bit_num-th bit calculation
      for bit_num in range(0, self.kbit):
        bound_sum = bound_sum + 0.5**(self.kbit - bit_num)
        bound_bit[self.kbit - 1 - bit_num] = bound_sum
      bound_bit = bound_bit.cuda()
      mykey = "k"
      if self.six_sigma.get(mykey) is None:
        std, _ = torch.std_mean(key)
        alpha = 5 * std
        self.six_sigma[mykey] = alpha
        print("dict key: ", mykey, "Added Alpha: ", alpha)
      else:
        alpha = self.six_sigma.get(mykey)

      k_clamp = key.clamp(-alpha, alpha)
      for bit_num in range(1, self.kbit + 1):
        k_quant = self.quantize_by_bit(
            k_clamp, bit_num=bit_num, alpha_key=mykey, ori_bit_num=self.kbit)
        k_clamp -= k_quant
        # new_attention_weights: [8, 20, 1024, 1024]
        # my_actual_mask: [1, 1, 1024, 1024]
        new_attention_weights += torch.matmul(query, k_quant.transpose(-2, -1))
        if not self.is_cross_attention:
          query_length, key_length = query.size(-2), k_quant.size(-2)
          causal_mask = self.bias[:, :, key_length -
                                  query_length:key_length, :key_length].bool()
          new_attention_weights = torch.where(
              causal_mask, new_attention_weights,
              self.masked_bias.to(new_attention_weights.dtype))
        # print("ISCAREMOVE: new_attention_weights Size: ",
        #       new_attention_weights.size())
        # print("ISCAREMOVE: my_actual_mask Size: ",
        #       my_actual_mask.size())

        for i in range(0, new_attention_weights.size(0)):
          for j in range(0, new_attention_weights.size(1)):
            # Kmax * Qsum
            bound = bound_bit[bit_num - 1] * (self.six_sigma[mykey] /
                                              bound_bit[0]) * q_abs_sum[i, j, :]
            bound = bound.unsqueeze(-1)
            bound = bound.repeat(1, new_attention_weights.size(3))
            array = new_attention_weights[i, j, :, :]
            # array size: [1024, 1024]
            # print("ISCAREMOVE: array Size: ", array.size())
            new_array = self.soft_thres_layer(array + bound)
            # Index of small (pruned out) values.
            out_ind = torch.where(new_array < -1e4 + 1)
            array[out_ind] = -1e4
            new_attention_weights[i, j, :, :] = array
          # row: [20, 1024, 1024]
          # my_actual_mask: [1, 1, 1024, 1024]
          row = new_attention_weights[i]
          # print(f"ROW SIZE: {row.size()}")
          # print(f"Actual MASK SIZE: {my_actual_mask.size()}")
          non_sparsity = (row > -1e4 + 1).sum() / (
              (my_actual_mask[0, :, :, :] > -1e4 + 1).sum() * row.size(0))
          sparsity = (1 - non_sparsity)
          # print('non sparsity: ', sparsity)
          # print('row size: ', row.size())
          # print('my_actual_mask size: ', my_actual_mask[0, :, :, :].size())
          assert sparsity >= 0, "Sparsity can not be negative!"
          var += ((my_actual_mask[0, :, :, :] > -1e4 + 1).sum() * row.size(0) -
                  sigmoid(100 * (row + 1e4 - 1)).sum()) / (
                      (my_actual_mask[0, :, :, :] > -1e4 + 1).sum() *
                      row.size(0)) * 100
          self.sparsity[bit_num - 1] += sparsity.item()
      self.sparsity = [i / new_attention_weights.size(0) for i in self.sparsity]
      attn_weights = new_attention_weights
      if self.scale_attn_weights:
        attn_weights = attn_weights / (value.size(-1)**0.5)
    elif self.prun and not self.early_stop:
    ##### Mingu added for RRAM #####
      mkey = 'q'
      q_rram = self.quantize(query,  bit_num=4, alpha_key = mkey)
      # print(f'max q diff {torch.max(torch.abs(query_layer - q))}, q mean : {torch.mean(query_layer)}')
      mkey = 'k'
      k_rram = self.quantize(key, bit_num=8, alpha_key = mkey)
      # print(f'max k diff {torch.max(torch.abs(key_layer - k))}, k mean : {torch.mean(key_layer)}')
      attention_scores_rram = torch.matmul(q_rram, k_rram.transpose(-1, -2))

      mkey = 'scores'
      attention_scores_rram = self.quantize(attention_scores_rram,  bit_num=5, alpha_key = mkey)
#       attention_scores_rram = attention_scores_rram + attention_mask
#       attention_scores_rram = attention_scores_rram + attention_mask.transpose(2,3)
      if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        # AMIR-ISCA: This is running! Not cross attention!
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length -
                                query_length:key_length, :key_length].bool()
        attention_scores_rram = torch.where(causal_mask, attention_scores_rram,
                                  self.masked_bias.to(attention_scores_rram.dtype))
      if attention_mask is not None:
        assert torch.count_nonzero(attention_mask) == 0, "All are zero!"
        # amir
        attention_mask[attention_mask == -10000] = -10000
        # rima
        # Apply the attention mask: [Batch, 1, 1, 1024]
        attention_scores_rram = attention_scores_rram + attention_mask
    #################################
      new_attention_weights = torch.zeros(attn_weights.size()).cuda()
      # Iterate over each batch.
      for i in range(0, attn_weights.size(0)):
        # Row Size: (20, 1024, 1024)
        row = attn_weights[i, :]
        # new_row = self.soft_thres_layer(row)  commented for rram

        if USE_ISCA_PRUN:
          new_row = self.soft_thres_layer(row)
        else:
          new_row = row.clone()
          rram_row = attention_scores_rram[i,:]
          thres_quant = self.soft_thres_layer.alpha
          if USE_RRAM:
            new_row[rram_row < thres_quant] = -1e4
          else:
            new_row[row < thres_quant] = -1e4
        new_attention_weights[i, :] = new_row
        var += (
            (my_actual_mask[0, :, :, :] > -1e4 + 1).sum() * new_row.size(0) -
            sigmoid(100 * (new_row + 1e4 - 1)).sum()) / (
                (my_actual_mask[0, :, :, :] > -1e4 + 1).sum() *
                new_row.size(0)) * 100
      non_sparsity = (new_attention_weights > -1e4 + 1).sum() / (
          (my_actual_mask > -1e4 + 1).sum() * new_attention_weights.size(0) *
          new_attention_weights.size(1))
      sparsity = (1 - non_sparsity)
      attn_weights = new_attention_weights
      if self.scale_attn_weights:
        assert self.scale_attn_weights, 'not scaling weights'
        attn_weights = attn_weights / (value.size(-1)**0.5)

     #----------Mingu Added——————————————————————————————————————
      if attention_mask is not None:
        pct = PCT # This pct means memory size = pct(0.3) * sequence size
        unmasked_cnt = 0

        prun_val = 10000 / (value.size(-1)**0.5)

        # for i in range(0, my_actual_mask.size(0)):
        #     unmasked_cnt = unmasked_cnt + (my_actual_mask[i,:,:,:] != -10000).sum() * (my_actual_mask[i,:,:,:] != -10000).sum()
        for i in range(0, my_actual_mask.size(0)):
            unmasked_cnt = unmasked_cnt + (my_actual_mask[i,:,:,:] != -10000).sum()
        # mask = [batch, 1, 1, s]
        unmasked_cnt = unmasked_cnt * attn_weights.size(0) * attn_weights.size(1)  # head and batch number mulplied
        # mask = [batch, 1, 1, s]
        # unmasked_cnt = unmasked_cnt * attn_weights.size(1)  # head number mulplied
        # print('total', attention_scores.numel(), 'unmasked', unmasked_cnt, 'less than',  (attention_scores > -prun_val).sum() )
        # print('min', attention_scores.min(), 'prun val', -prun_val)
        # 2D mask case
        sparsity    = (   unmasked_cnt - (attn_weights > -prun_val).sum()  ) / (unmasked_cnt)*100
        # # print(attention_scores.min(), attention_scores.max(),-prun_val)
        # print(unmasked_cnt ,(attention_scores > -prun_val).sum() )
        # print(f'new sparsity : {sparsity}')
        unprun_max  = torch.max(torch.sum((attn_weights > -prun_val), 3))
        # unprun_avg  = torch.mean(torch.sum((attention_scores > -prun_val), 3).float())
        unprun_avg_dim1_2 = torch.mean(   torch.sum((attn_weights > -prun_val), 3).float(), [0,1]  )
        # Apr 14 ----
        unpruned_pos = torch.where(attn_weights > -prun_val)
        mod_4 = torch.remainder((unpruned_pos[3] + 1), 4)
        mod_2 = torch.remainder((unpruned_pos[3] + 1), 2)
        core4 = torch.unique(mod_4, return_counts = True)[1]
        core2 = torch.unique(mod_2, return_counts = True)[1]
        minmax_mod2 = max(core2) / min(core2)
        delay_mod2 = max(core2) * 2 / sum(core2)
        minmax_mod4 = max(core4) / min(core4)
        delay_mod4 = max(core4) * 4 / sum(core4)
        # Please comment out this print line after the issue is found
        # print(f'minmax_mod 2 {minmax_mod2}, delay mod 2 {delay_mod2}, minmax mod 4 {minmax_mod4}, delay mod 4{delay_mod4}')
        s = math.sqrt(unmasked_cnt / (attn_weights.size(0) * attn_weights.size(1)))

        # print('s 1/4', s_quarter)
        quarter1 = len(torch.where((unpruned_pos[3] + 1) / s < (1 / 4))[0])
        quarter2 = len(torch.where((unpruned_pos[3] + 1 )/ s< 2 * (1/ 4))[0]) - quarter1
        quarter3 = len(torch.where((unpruned_pos[3] + 1) / s< 3 * (1 / 4))[0]) - (quarter2 + quarter1)
        quarter4 = len(torch.where((unpruned_pos[3] + 1 )/ s< 4 * (1 / 4))[0]) - (quarter2 + quarter1 + quarter3)
        # print(quarter1, quarter2,quarter3, quarter4)
        # minmax_seq2 = max((quarter1 + quarter2), (quarter3 + quarter4)) / (min((quarter1 + quarter2), (quarter3 + quarter4))) # add 1 to prevent zero devision
        minmax_seq2 = max((quarter1 + quarter2), (quarter3 + quarter4)) / (min((quarter1 + quarter2), (quarter3 + quarter4)) + 1)
        # minmax_seq2 =1
        delay_seq2 =  max((quarter1 + quarter2), (quarter3 + quarter4)) * 2 / (quarter1 + quarter2 + quarter3 + quarter4)
        # delay_seq2 = 1
        minmax_seq4 = max(quarter1, quarter2, quarter3, quarter4) / (min(quarter1, quarter2, quarter3, quarter4) + 1)
        delay_seq4 =  max(quarter1, quarter2, quarter3, quarter4)  * 4 / (quarter1 + quarter2 + quarter3 + quarter4)
        # print('minmaxseq2', minmax_seq2, 'minmax seq 4 ', minmax_seq4)
        # print('delay seq2, ', delay_seq2, 'delay seq4 ', delay_seq4)
        # -------------
        unprun_avg = torch.mean(unprun_avg_dim1_2[unprun_avg_dim1_2>0])
        avg_unmasked_pct = unmasked_cnt/torch.numel(attn_weights)
        unprun_ov_pct = (torch.sum((attn_weights > -prun_val), 3)>attn_weights.size(3)*pct).sum() / (attn_weights.size(0)*attn_weights.size(1)*attn_weights.size(2)) # probablity that unpruned K number is more than memory capacity


        ###############  Calculation for the rows > pct% was survived ###############
        idx_ov_pct = (torch.sum((attn_weights > -prun_val), 3)>attn_weights.size(3)*pct)  ## all the indices of memory over cases
        attention_scores_ov_pct = torch.zeros_like(attn_weights) - prun_val   ## all element is reset to be pruned value
        attention_scores_ov_pct[idx_ov_pct,:] = attn_weights[idx_ov_pct,:]      ## copy only the row where more than 30% is unpruned

        corr_ov_pct = (attention_scores_ov_pct > -prun_val).float()                        ## unpruned part becomes 1
        common_ov_pct = torch.logical_and(corr_ov_pct[:,:,1:corr_ov_pct.size(2)-1,:], corr_ov_pct[:,:,0:corr_ov_pct.size(2)-2,:])
        ## above line make 1 for the idx where 2 adjects rows have a common unpruned <- means the vector K / V can be reused
        common_ov_pct_sum = torch.sum(common_ov_pct,3).float()    ## how many can be reused
        common_ov_pct_sum[common_ov_pct_sum > attn_weights.size(3)*pct] = attn_weights.size(3)*pct
        ## Note due to memory limit, maximum reuse is limited by 30%

        new_fetch_avg_ov_pct = 0
        if (idx_ov_pct.sum()>0):
            new_fetch_avg_ov_pct = torch.mean(  torch.sum(corr_ov_pct, 3).float()  )  ## >pct case's total new read number

        new_fetch_avg_ov_pct = new_fetch_avg_ov_pct - torch.mean(common_ov_pct_sum)  ## K / V reuse portion should be subtracted
        ############################################################################


        ######################  Calculation for the other rows  ######################
        #corr[batch, head, q_idx, k_idx]
        corr = (attn_weights > -prun_val).float()       ## unpruned part becomes 1
        corr[idx_ov_pct, :] = 0  ## if more than pct% chosen, exclude in this calculation
        #corr = corr.float() - (attention_mask/1000)*2       ## makes 2 for masked part

        # new_read = (corr[:,:,1:corr.size(2)-1,:] - corr[:,:,0:corr.size(2)-2,:])>0

        new_read = (corr[:,:,1:corr.size(2)-1,:] - corr[:,:,0:corr.size(2)-2,:])>0
        new_fetch_max = torch.max(   torch.sum(new_read,3))
            # below counts the extra mem slot after storing unpruned K / Vs
        unused_mem_slots = corr.size(3)*pct - torch.sum(corr[:,:,0:corr.size(2)-2,:], 3).float()

        # unused storage space might have the contents for new read. This probablity should be considered
        # below line shows the probablity that a certain "new read" content is in the un-used slot coincidentally.
        # print(unused_mem_slots.size(), (torch.sum((my_actual_mask>-9999),3)).repeat(1,new_read.size(1),1).size())
        reuse_prob = unused_mem_slots / (torch.sum((my_actual_mask[:,:,0:my_actual_mask.size(2)-2,:]>-9999),3)).repeat(1,unused_mem_slots.size(1),1)
        reuse_prob[reuse_prob>1] = 1.0  # probability cannot be >1
        new_fetch_max = torch.max(   torch.sum(new_read,3))

        if (REUSE_BY_UNUSED_SLOT_FLAG == 1):
            new_read_sum = torch.sum(new_read,3)* (1- reuse_prob)  # reuse portion considered
        else:
            new_read_sum = torch.sum(new_read,3)

        new_fetch_max = torch.max(   torch.sum(new_read,3))

        ######################  Calculation for intersection  ######################
        ### for row N: mem over case, for row N+1: normal case ###
        ### in this case, row N's contents can be reused in the N+1'th row
        unprun_num_if_ov_pct = 1
        if (idx_ov_pct.sum()>0):
            unprun_num_if_ov_pct = corr_ov_pct.sum() / idx_ov_pct.sum()  # how many tokens are unpruned if they are >30% case
        prob_to_present = ( attn_weights.size(3)*pct / unprun_num_if_ov_pct ) # prob of the vector to be reused in N+1 is present in on-chip memory

        ## reuse case when N: mem over case, and N+1: normal case
        reuse_intsec_ov_norm = torch.mean(   torch.sum(  torch.logical_and(corr[:,:,1:corr.size(2)-1,:], corr_ov_pct[:,:,0:corr.size(2)-2,:]).float(), 3)  )

        ## reuse case when N: normal case, and N+1: mem over case
        reuse_intsec_norm_ov = torch.mean(   torch.sum(  torch.logical_and(corr_ov_pct[:,:,1:corr.size(2)-1,:], corr[:,:,0:corr.size(2)-2,:]).float(), 3)  )

        ###################### Total calculation ####################
        # new_fetch_avg = torch.mean(  torch.sum(new_read,3).float() ) + new_fetch_avg_ov_pct - reuse_intsec_ov_norm * prob_to_present - reuse_intsec_norm_ov
        new_fetch_avg = torch.mean(  new_read_sum ) + new_fetch_avg_ov_pct - reuse_intsec_ov_norm * prob_to_present - reuse_intsec_norm_ov                  # in above line, 1st term: non ov_pct case <- this term consider reuse case internally
            #                2nd term: ov_pct case     <- this term considers reuse case within ov_pct cases
            #                3rd term / 4 terms: intersection of above cases
        ##############################################################################
    # rime
    # print("Var: ", var)
    # print("Sparsity: ", sparsity)
    if self.quant:
            attn_weights = self.quantize(attn_weights, bit_num=12 ,alpha_key='scores')
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    if self.quant:
            attn_weights = self.quantize(attn_weights, bit_num=8 ,alpha_key='scores_softmax')
    if self.quant:
            value = self.quantize(value, bit_num=8, alpha_key='v')
    # Downcast (if necessary) back to V's dtype (if in mixed-precision)
    # -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
      attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    if self.quant:
      attn_output = self.quantize(attn_output, bit_num=16, alpha_key='out')
    if self.prun and self.early_stop:
      return attn_output, attn_weights, var, sparsity
    elif self.prun and not self.early_stop:
      return attn_output, attn_weights, var, sparsity, unprun_avg, new_fetch_avg, unprun_ov_pct, avg_unmasked_pct, minmax_mod2, delay_mod2, minmax_mod4, delay_mod4, minmax_seq2, delay_seq2,  minmax_seq4, delay_seq4
    else:
      return attn_output, attn_weights, 0, 0
    return attn_output, attn_weights, 0, 0

  def _upcast_and_reordered_attn(self,
                                 query,
                                 key,
                                 value,
                                 attention_mask=None,
                                 head_mask=None):
    # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
    bsz, num_heads, q_seq_len, dk = query.size()
    _, _, k_seq_len, _ = key.size()

    # Preallocate attn_weights for `baddbmm`
    attn_weights = torch.empty(
        bsz * num_heads,
        q_seq_len,
        k_seq_len,
        dtype=torch.float32,
        device=query.device)

    # Compute Scale Factor
    scale_factor = 1.0
    if self.scale_attn_weights:
      scale_factor /= float(value.size(-1))**0.5

    if self.scale_attn_by_inverse_layer_idx:
      scale_factor /= float(self.layer_idx + 1)

    # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
    if is_amp_available:
      with autocast(enabled=False):
        q, k = query.reshape(-1, q_seq_len,
                             dk), key.transpose(-1,
                                                -2).reshape(-1, dk, k_seq_len)
        attn_weights = torch.baddbmm(
            attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
        attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len,
                                            k_seq_len)
    else:
      q, k = query.reshape(-1, q_seq_len,
                           dk), key.transpose(-1,
                                              -2).reshape(-1, dk, k_seq_len)
      attn_weights = torch.baddbmm(
          attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
      attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

    if not self.is_cross_attention:
      # if only "normal" attention layer implements causal mask
      query_length, key_length = query.size(-2), key.size(-2)
      causal_mask = self.bias[:, :, key_length -
                              query_length:key_length, :key_length].bool()
      attn_weights = torch.where(causal_mask, attn_weights,
                                 self.masked_bias.to(attn_weights.dtype))

    if attention_mask is not None:
      # Apply the attention mask
      attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
    if attn_weights.dtype != torch.float32:
      raise RuntimeError(
          "Error with upcasting, attn_weights does not have dtype torch.float32"
      )
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
      attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights

  def _split_heads(self, tensor, num_heads, attn_head_size):
    """
        Splits hidden_size dim into attn_head_size and num_heads
        """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1,
                          3)  # (batch, head, seq_length, head_features)

  def _merge_heads(self, tensor, num_heads, attn_head_size):
    """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)

  def forward(
      self,
      hidden_states,
      layer_past=None,
      attention_mask=None,
      head_mask=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      use_cache=False,
      output_attentions=False,
  ):
    if encoder_hidden_states is not None:
      if not hasattr(self, "q_attn"):
        raise ValueError(
            "If class is used as cross attention, the weights `q_attn` have to be defined. "
            "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
        )

      query = self.q_attn(hidden_states)
      key, value = self.c_attn(encoder_hidden_states).split(
          self.split_size, dim=2)
      attention_mask = encoder_attention_mask
    else:
      query, key, value = self.c_attn(hidden_states).split(
          self.split_size, dim=2)

    # GPT2-LARGE:
    # Before: [Batch, 1024, 1280]
    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)
    # After: [Batch, 20, 1024, 64]

    if layer_past is not None:
      assert False, "ISCA: Oh No! Not calling this path!"
      past_key, past_value = layer_past
      key = torch.cat((past_key, key), dim=-2)
      value = torch.cat((past_value, value), dim=-2)

    if use_cache:
      present = (key, value)
    else:
      present = None

    if self.reorder_and_upcast_attn:
      assert False, "AMIR: Oh no! This path not working!"
      attn_output, attn_weights = self._upcast_and_reordered_attn(
          query, key, value, attention_mask, head_mask)
    else:
      attn_output, attn_weights, var, sparsity, unprun_avg, new_fetch_avg, unprun_ov_pct, avg_unmasked_pct, minmax_mod2, delay_mod2, minmax_mod4, delay_mod4, minmax_seq2, delay_seq2,  minmax_seq4, delay_seq4 = self._attn(
          query, key, value, attention_mask, head_mask)

    attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
      outputs += (attn_weights,)

    if self.prun:
      return outputs, var, sparsity, unprun_avg, new_fetch_avg, unprun_ov_pct, avg_unmasked_pct, minmax_mod2, delay_mod2, minmax_mod4, delay_mod4, minmax_seq2, delay_seq2,  minmax_seq4, delay_seq4
    else:
      return outputs, 0, 0

    return outputs, 0, 0  # a, present, (attentions)


class GPT2MLP(nn.Module):

  def __init__(self, intermediate_size, config):
    super().__init__()
    embed_dim = config.hidden_size
    self.c_fc = Conv1D(intermediate_size, embed_dim)
    self.c_proj = Conv1D(embed_dim, intermediate_size)
    self.act = ACT2FN[config.activation_function]
    self.dropout = nn.Dropout(config.resid_pdrop)

  def forward(self, hidden_states):
    hidden_states = self.c_fc(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.c_proj(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


class GPT2Block(nn.Module):

  def __init__(self, config, layer_idx=None):
    super().__init__()
    hidden_size = config.hidden_size
    inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

    self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
    self.attn = GPT2Attention(config, layer_idx=layer_idx)
    self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    if config.add_cross_attention:
      self.crossattention = GPT2Attention(config, is_cross_attention=True)
      self.ln_cross_attn = nn.LayerNorm(
          hidden_size, eps=config.layer_norm_epsilon)

    self.mlp = GPT2MLP(inner_dim, config)

  def forward(
      self,
      hidden_states,
      layer_past=None,
      attention_mask=None,
      head_mask=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      use_cache=False,
      output_attentions=False,
  ):
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    # amir
    attn_outputs, total_var, total_sparsity, total_unprun_avg, total_new_fetch_avg, total_unprun_ov_pct, total_avg_unmasked_pct, total_minmax_mod2, total_delay_mod2, total_minmax_mod4, total_delay_mod4, total_minmax_seq2, total_delay_seq2, total_minmax_seq4, total_delay_seq4 = self.attn(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    # rima
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]
    # residual connection
    hidden_states = attn_output + residual

    if encoder_hidden_states is not None:
      # add one self-attention block for cross-attention
      if not hasattr(self, "crossattention"):
        raise ValueError(
            f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
            "cross-attention layers by setting `config.add_cross_attention=True`"
        )
      residual = hidden_states
      hidden_states = self.ln_cross_attn(hidden_states)
      cross_attn_outputs, var, sparsity, unprun_avg, new_fetch_avg, unprun_ov_pct, avg_unmasked_pct, minmax_mod2, delay_mod2, minmax_mod4, delay_mod4, minmax_seq2, delay_seq2,  minmax_seq4, delay_seq4 = self.crossattention(
          hidden_states,
          attention_mask=attention_mask,
          head_mask=head_mask,
          encoder_hidden_states=encoder_hidden_states,
          encoder_attention_mask=encoder_attention_mask,
          output_attentions=output_attentions,
      )
      attn_output = cross_attn_outputs[0]
      # residual connection
      hidden_states = residual + attn_output
      outputs = outputs + cross_attn_outputs[
          2:]  # add cross attentions if we output attention weights
      total_var += var
      total_sparsity += sparsity
      total_unprun_avg += unprun_avg
      total_new_fetch_avg += new_fetch_avg
      total_unprun_ov_pct += unprun_ov_pct
      total_avg_unmasked_pct += avg_unmasked_pct
      total_minmax_mod2 += minmax_mod2
      total_minmax_mod4 += minmax_mod4
      total_delay_mod2 += delay_mod2
      total_delay_mod4 += delay_mod4
      total_minmax_seq2 += minmax_seq2
      total_minmax_seq4 += minmax_seq4
      total_delay_seq2 += delay_seq2
      total_delay_seq4 += delay_seq4
    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    if use_cache:
      outputs = (hidden_states,) + outputs
    else:
      outputs = (hidden_states,) + outputs[1:]
    # hidden_states, present, (attentions, cross_attentions)
    return outputs, total_var, total_sparsity, total_unprun_avg, total_new_fetch_avg, total_unprun_ov_pct, total_avg_unmasked_pct, total_minmax_mod2, total_delay_mod2, total_minmax_mod4, total_delay_mod4, total_minmax_seq2, total_delay_seq2, total_minmax_seq4, total_delay_seq4


class GPT2PreTrainedModel(PreTrainedModel):
  """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained
    models.
    """

  config_class = GPT2Config
  load_tf_weights = load_tf_weights_in_gpt2
  base_model_prefix = "transformer"
  is_parallelizable = True
  supports_gradient_checkpointing = True

  def __init__(self, *inputs, **kwargs):
    print("Amir load!")
    super().__init__(*inputs, **kwargs)

  def _init_weights(self, module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, Conv1D)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
      if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)

    # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
    #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
    #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
    #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
    #
    # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
    for name, p in module.named_parameters():
      if "c_proj" in name and "weight" in name:
        # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
        p.data.normal_(
            mean=0.0,
            std=(self.config.initializer_range /
                 math.sqrt(2 * self.config.n_layer)))

  def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, GPT2Model):
      module.gradient_checkpointing = value


@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
  """
    Base class for outputs of models predicting if two sentences are consecutive
    or not.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when
          `labels` is provided): Language modeling loss.
        mc_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when
          `mc_labels` is provided): Multiple choice classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices,
          sequence_length, config.vocab_size)`): Prediction scores of the
          language modeling head (scores for each vocabulary token before
          SoftMax).
        mc_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
          Prediction scores of the multiple choice classification head (scores
          for each choice before SoftMax).
        past_key_values (`Tuple[Tuple[torch.Tensor]]`, *optional*, returned when
          `use_cache=True` is passed or when `config.use_cache=True`): Tuple of
          length `config.n_layers`, containing tuples of tensors of shape
          `(batch_size, num_heads, sequence_length, embed_size_per_head)`).
          Contains pre-computed hidden-states (key and values in the attention
          blocks) that can be used (see `past_key_values` input) to speed up
          sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when
          `output_hidden_states=True` is passed or when
          `config.output_hidden_states=True`): Tuple of `torch.FloatTensor` (one
          for the output of the embeddings + one for the output of each layer)
          of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states
          of the model at the output of each layer plus the initial embedding
          outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when
          `output_attentions=True` is passed or when
          `config.output_attentions=True`): Tuple of `torch.FloatTensor` (one
          for each layer) of shape `(batch_size, num_heads, sequence_length,
          sequence_length)`.  GPT2Attentions weights after the attention
          softmax, used to compute the weighted average in the self-attention
          heads.
  """

  loss: Optional[torch.FloatTensor] = None
  mc_loss: Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  mc_logits: torch.FloatTensor = None
  past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  attentions: Optional[Tuple[torch.FloatTensor]] = None


GPT2_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPT2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`GPT2Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:

                - gpt2: 12
                - gpt2-medium: 24
                - gpt2-large: 36
                - gpt2-xl: 48

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl, which has a total of 48 attention modules:
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with gpt2-large:
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
  _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

  def __init__(self, config):
    super().__init__(config)

    self.embed_dim = config.hidden_size

    self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
    self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

    self.drop = nn.Dropout(config.embd_pdrop)
    self.h = nn.ModuleList([
        GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)
    ])
    self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    # Model parallel
    self.model_parallel = False
    self.device_map = None
    self.gradient_checkpointing = False

    # Initialize weights and apply final processing
    self.post_init()

  @add_start_docstrings(PARALLELIZE_DOCSTRING)
  def parallelize(self, device_map=None):
    # Check validity of device_map
    self.device_map = (
        get_device_map(len(self.h), range(torch.cuda.device_count()))
        if device_map is None else device_map)
    assert_device_map(self.device_map, len(self.h))
    self.model_parallel = True
    self.first_device = "cpu" if "cpu" in self.device_map.keys(
    ) else "cuda:" + str(min(self.device_map.keys()))
    self.last_device = "cuda:" + str(max(self.device_map.keys()))
    self.wte = self.wte.to(self.first_device)
    self.wpe = self.wpe.to(self.first_device)
    # Load onto devices
    for k, v in self.device_map.items():
      for block in v:
        cuda_device = "cuda:" + str(k)
        self.h[block] = self.h[block].to(cuda_device)
    # ln_f to last
    self.ln_f = self.ln_f.to(self.last_device)

  @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
  def deparallelize(self):
    self.model_parallel = False
    self.device_map = None
    self.first_device = "cpu"
    self.last_device = "cpu"
    self.wte = self.wte.to("cpu")
    self.wpe = self.wpe.to("cpu")
    for index in range(len(self.h)):
      self.h[index] = self.h[index].to("cpu")
    self.ln_f = self.ln_f.to("cpu")
    torch.cuda.empty_cache()

  def get_input_embeddings(self):
    return self.wte

  def set_input_embeddings(self, new_embeddings):
    self.wte = new_embeddings

  def _prune_heads(self, heads_to_prune):
    """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of
        heads to prune in this layer}
        """
    for layer, heads in heads_to_prune.items():
      self.h[layer].attn.prune_heads(heads)

  @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
  @add_code_sample_docstrings(
      processor_class=_TOKENIZER_FOR_DOC,
      checkpoint=_CHECKPOINT_FOR_DOC,
      output_type=BaseModelOutputWithPastAndCrossAttentions,
      config_class=_CONFIG_FOR_DOC,
  )
  def forward(
      self,
      input_ids=None,
      past_key_values=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):

    # amir
    total_var = 0
    total_sparsity = 0
    # Zheng added -----------------
    self.total_unprun_avg = 0
    self.total_new_fetch_avg = 0
    self.total_unprun_ov_pct = 0
    self.total_avg_unmasked_pct = 0
    self.total_cnt = 0
    self.overlap_first = 0
    self.overlap_last = 0
    self.unprun_first = 0
    self.unprun_last = 0


    self.total_minmax_mod2 = 0
    self.total_delay_mod2 = 0
    self.total_minmax_mod4 = 0
    self.total_delay_mod4 = 0
    self.total_minmax_seq2 = 0
    self.total_delay_seq2 = 0
    self.total_minmax_seq4 = 0
    self.total_delay_seq4 = 0
    # ---------------


    if self.config.early_stopping:
      total_sparsity = [0 for _ in range(KBIT)]
    # rima

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
      raise ValueError(
          "You cannot specify both input_ids and inputs_embeds at the same time"
      )
    elif input_ids is not None:
      input_shape = input_ids.size()
      input_ids = input_ids.view(-1, input_shape[-1])
      batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.size()[:-1]
      batch_size = inputs_embeds.shape[0]
    else:
      raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
      token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
      position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
      past_length = 0
      past_key_values = tuple([None] * len(self.h))
    else:
      past_length = past_key_values[0][0].size(-2)
    if position_ids is None:
      position_ids = torch.arange(
          past_length,
          input_shape[-1] + past_length,
          dtype=torch.long,
          device=device)
      position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # GPT2Attention mask.
    if attention_mask is not None:
      if batch_size <= 0:
        raise ValueError("batch_size has to be defined and > 0")
      attention_mask = attention_mask.view(batch_size, -1)
      # We create a 3D attention mask from a 2D tensor mask.
      # Sizes are [batch_size, 1, 1, to_seq_length]
      # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
      # this attention mask is more simple than the triangular masking of causal attention
      # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
      attention_mask = attention_mask[:, None, None, :]

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
      attention_mask = (1.0 - attention_mask) * -10000.0

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.add_cross_attention and encoder_hidden_states is not None:
      encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
      )
      encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
      if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
      encoder_attention_mask = self.invert_attention_mask(
          encoder_attention_mask)
    else:
      encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
      inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    if token_type_ids is not None:
      token_type_embeds = self.wte(token_type_ids)
      hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = (
    ) if output_attentions and self.config.add_cross_attention else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
      # Model parallel
      if self.model_parallel:
        torch.cuda.set_device(hidden_states.device)
        # Ensure layer_past is on same device as hidden_states (might not be correct)
        if layer_past is not None:
          layer_past = tuple(
              past_state.to(hidden_states.device) for past_state in layer_past)
        # Ensure that attention_mask is always on the same device as hidden_states
        if attention_mask is not None:
          attention_mask = attention_mask.to(hidden_states.device)
        if isinstance(head_mask, torch.Tensor):
          head_mask = head_mask.to(hidden_states.device)
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      if self.gradient_checkpointing and self.training:
        assert False, "Oh No! This path is not working!"
        if use_cache:
          logger.warning(
              "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
          )
          use_cache = False

        def create_custom_forward(module):

          def custom_forward(*inputs):
            # None for past_key_value
            return module(*inputs, use_cache, output_attentions)

          return custom_forward

        outputs, var, sparsity = torch.utils.checkpoint.checkpoint(
            create_custom_forward(block),
            hidden_states,
            None,
            attention_mask,
            head_mask[i],
            encoder_hidden_states,
            encoder_attention_mask,
        )
      else:
        outputs, var, sparsity, unprun_avg, new_fetch_avg, unprun_ov_pct, avg_unmasked_pct, minmax_mod2, delay_mod2, minmax_mod4, delay_mod4, minmax_seq2, delay_seq2,  minmax_seq4, delay_seq4 = block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask[i],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

      hidden_states = outputs[0]
      #Zheng added =------------------
      self.overlap_first = 0
      self.overlap_last = 0
      self.k_first = 0
      self.k_last = 0

      self.total_unprun_avg += unprun_avg
      self.total_new_fetch_avg += new_fetch_avg
      self.total_unprun_ov_pct += unprun_ov_pct
      self.total_avg_unmasked_pct += avg_unmasked_pct

      self.total_minmax_mod2 += minmax_mod2
      self.total_delay_mod2 += delay_mod2
      self.total_minmax_mod4 += minmax_mod4
      self.total_delay_mod4 += delay_mod4
      self.total_minmax_seq2 += minmax_seq2
      self.total_delay_seq2 += delay_seq2
      self.total_minmax_seq4 += minmax_seq4
      self.total_delay_seq4 += delay_seq4

      
      if i == 0:
        print(f'unprun_avg = {unprun_avg}, new_fetch_avg = {new_fetch_avg}')
        self.overlap_first += unprun_avg - new_fetch_avg
        print(f'curretn overlap 1st {self.overlap_first}')
        self.k_first += unprun_avg
      elif i == len(self.h) - 1:
          print(f'unprun_avg = {unprun_avg}, new_fetch_avg = {new_fetch_avg}')
          self.overlap_last += unprun_avg - new_fetch_avg
          self.k_last += unprun_avg
          print(f'curretn overlap last {self.overlap_last}')
      self.total_cnt += 1

      if (self.total_cnt % 5 == 0):
        #print("cnt",  self.total_cnt)
        print("avg # of new fetch", self.total_new_fetch_avg / self.total_cnt)
        print("avg # of unpruned K", self.total_unprun_avg / self.total_cnt)
        print("avg # of cases with more # of unpruned K than mem cap", self.total_unprun_ov_pct / self.total_cnt)
        print("avg portion of unmasked part", self.total_avg_unmasked_pct / self.total_cnt)
        print('avg overlap 1st layer : ', len(self.h) * self.overlap_first / self.total_cnt)
        print('avg overlap last layer : ', len(self.h) * self.overlap_last / self.total_cnt)
        print('raw overlap 1st : ', self.overlap_first)
        print('unpruned k first layer : ', len(self.h) * self.k_first / self.total_cnt)
        print('unpruned k last layer : ', len(self.h) * self.k_last / self.total_cnt)
        # print('total core2,', self.total_core2 / self.total_cnt)
        # print('total core4, ', self.total_core4 / self.total_cnt)
        print('minmax ratio mod 2, ', self.total_minmax_mod2 / self.total_cnt)
        print('minmax ratio mod 4, ', self.total_minmax_mod4 / self.total_cnt)
        print('minmax ratio seq 2, ', self.total_minmax_seq2 / self.total_cnt)
        print('minmax ratio seq 4, ', self.total_minmax_seq4 / self.total_cnt)
        print('delay mod 2, ', self.total_delay_mod2 / self.total_cnt)
        print('delay mod 4, ', self.total_delay_mod4 / self.total_cnt)
        print('delay seq 2, ', self.total_delay_seq2 / self.total_cnt)
        print('delay seq 4, ', self.total_delay_seq4 / self.total_cnt)
      # -----------------
      # amir
      total_var += var
      if not self.config.early_stopping:
        total_sparsity += sparsity
      else:
        for kb in range(KBIT):
          total_sparsity[kb] += sparsity[kb]
      # rima
      if use_cache:
        presents = presents + (outputs[1],)

      if output_attentions:
        all_self_attentions = all_self_attentions + (
            outputs[2 if use_cache else 1],)
        if self.config.add_cross_attention:
          all_cross_attentions = all_cross_attentions + (
              outputs[3 if use_cache else 2],)

      # Model Parallel: If it's the last layer for that device, put things on the next device
      if self.model_parallel:
        for k, v in self.device_map.items():
          if i == v[-1] and "cuda:" + str(k) != self.last_device:
            hidden_states = hidden_states.to("cuda:" + str(k + 1))

    # amir
    if self.config.early_stopping:
      total_sparsity = [i / len(self.h) for i in total_sparsity]
    # rima
    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    # amir
    if not self.config.early_stopping:
      total_sparsity = total_sparsity / len(self.h)
    # rima

    if not return_dict:
      return tuple(v for v in [
          hidden_states, presents, all_hidden_states, all_self_attentions,
          all_cross_attentions, total_var, total_sparsity
      ] if v is not None)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    ), total_var, total_sparsity


@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadModel(GPT2PreTrainedModel):
  _keys_to_ignore_on_load_missing = [
      r"attn.masked_bias", r"attn.bias", r"lm_head.weight"
  ]

  def __init__(self, config):
    super().__init__(config)
    self.transformer = GPT2Model(config)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # Model parallel
    self.model_parallel = False
    self.device_map = None

    # Initialize weights and apply final processing
    self.post_init()

    # amir
    if not self.config.early_stopping:
      self.sparsity = []
    else:
      self.sparsity = [[] for _ in range(KBIT)]
    # rima

  @add_start_docstrings(PARALLELIZE_DOCSTRING)
  def parallelize(self, device_map=None):
    self.device_map = (
        get_device_map(
            len(self.transformer.h), range(torch.cuda.device_count()))
        if device_map is None else device_map)
    assert_device_map(self.device_map, len(self.transformer.h))
    self.transformer.parallelize(self.device_map)
    self.lm_head = self.lm_head.to(self.transformer.first_device)
    self.model_parallel = True

  @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
  def deparallelize(self):
    self.transformer.deparallelize()
    self.transformer = self.transformer.to("cpu")
    self.lm_head = self.lm_head.to("cpu")
    self.model_parallel = False
    torch.cuda.empty_cache()

  def get_output_embeddings(self):
    return self.lm_head

  def set_output_embeddings(self, new_embeddings):
    self.lm_head = new_embeddings

  def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
      input_ids = input_ids[:, -1].unsqueeze(-1)
      if token_type_ids is not None:
        token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
      # create position_ids on the fly for batch generation
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)
      if past:
        position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
      position_ids = None
    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

  @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
  @add_code_sample_docstrings(
      processor_class=_TOKENIZER_FOR_DOC,
      checkpoint=_CHECKPOINT_FOR_DOC,
      output_type=CausalLMOutputWithCrossAttentions,
      config_class=_CONFIG_FOR_DOC,
  )
  def forward(
      self,
      input_ids=None,
      past_key_values=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    r"""

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`,
        *optional*):
            Labels for language modeling. Note that the labels **are shifted**
            inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ...,
            config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0,
            ..., config.vocab_size]`
        """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # amir
    transformer_outputs, var, sparsity = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    if not self.config.early_stopping:
      self.sparsity.append(sparsity)
    else:
      for i in range(KBIT):
        self.sparsity[i].append(sparsity[i])
    # rima
    hidden_states = transformer_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
      torch.cuda.set_device(self.transformer.first_device)
      hidden_states = hidden_states.to(self.lm_head.weight.device)

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      # Flatten the tokens
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(
          shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if not return_dict:
      print('Not return dict')
      output = (lm_logits,) + transformer_outputs[1:]
      return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithCrossAttentions(
        loss=loss + 1e-5 * (-var) if var else loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
        cross_attentions=transformer_outputs.cross_attentions,
    )

  @staticmethod
  def _reorder_cache(past: Tuple[Tuple[torch.Tensor]],
                     beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
    """
        This function is used to re-order the `past_key_values` cache if
        [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match
        `past_key_values` with the correct
        beam_idx at every generation step.
        """
    return tuple(
        tuple(
            past_state.index_select(0, beam_idx.to(past_state.device))
            for past_state in layer_past)
        for layer_past in past)


@add_start_docstrings(
    """
The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
input embeddings, the classification head takes as input the input of a specified classification token index in the
input sequence).
""",
    GPT2_START_DOCSTRING,
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
  _keys_to_ignore_on_load_missing = [
      r"attn.masked_bias", r"attn.bias", r"lm_head.weight"
  ]

  def __init__(self, config):
    super().__init__(config)
    config.num_labels = 1
    self.transformer = GPT2Model(config)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.multiple_choice_head = SequenceSummary(config)

    # Model parallel
    self.model_parallel = False
    self.device_map = None

    # Initialize weights and apply final processing
    self.post_init()

  @add_start_docstrings(PARALLELIZE_DOCSTRING)
  def parallelize(self, device_map=None):
    self.device_map = (
        get_device_map(
            len(self.transformer.h), range(torch.cuda.device_count()))
        if device_map is None else device_map)
    assert_device_map(self.device_map, len(self.transformer.h))
    self.transformer.parallelize(self.device_map)
    self.lm_head = self.lm_head.to(self.transformer.first_device)
    self.multiple_choice_head = self.multiple_choice_head.to(
        self.transformer.first_device)
    self.model_parallel = True

  @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
  def deparallelize(self):
    self.transformer.deparallelize()
    self.transformer = self.transformer.to("cpu")
    self.lm_head = self.lm_head.to("cpu")
    self.multiple_choice_head = self.multiple_choice_head.to("cpu")
    self.model_parallel = False
    torch.cuda.empty_cache()

  def get_output_embeddings(self):
    return self.lm_head

  def set_output_embeddings(self, new_embeddings):
    self.lm_head = new_embeddings

  def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
      input_ids = input_ids[:, -1].unsqueeze(-1)
      if token_type_ids is not None:
        token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
      # create position_ids on the fly for batch generation
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)
      if past:
        position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
      position_ids = None

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

  @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
  @replace_return_docstrings(
      output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
  def forward(
      self,
      input_ids=None,
      past_key_values=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      mc_token_ids=None,
      labels=None,
      mc_labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      **kwargs,
  ):
    r"""

        mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices)`,
        *optional*, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected
            in the range `[0, input_ids.size(-1) -
            1[`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`,
        *optional*):
            Labels for language modeling. Note that the labels **are shifted**
            inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ...,
            config.vocab_size - 1]` All labels set to
            `-100` are ignored (masked), the loss is only computed for labels in
            `[0, ..., config.vocab_size - 1]`
        mc_labels (`torch.LongTensor` of shape `(batch_size)`, *optional*):
            Labels for computing the multiple choice classification loss.
            Indices should be in `[0, ..., num_choices]`
            where *num_choices* is the size of the second dimension of the input
            tensors. (see *input_ids* above)

        Return:

        Example:

        ```python
        >>> import torch
        >>> from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> model = GPT2DoubleHeadsModel.from_pretrained("gpt2")

        >>> # Add a [CLS] to the vocabulary (we should train it also!)
        >>> num_added_tokens = tokenizer.add_special_tokens({"cls_token":
        "[CLS]"})

        >>> embedding_layer = model.resize_token_embeddings(
        ...     len(tokenizer)
        >>> )  # Update the model embeddings with the new vocabulary size

        >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute
        [CLS]"]
        >>> encoded_choices = [tokenizer.encode(s) for s in choices]
        >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for
        tokens in encoded_choices]

        >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch
        size: 1, number of choices: 2
        >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
        >>> lm_logits = outputs.logits
        >>> mc_logits = outputs.mc_logits
        ```
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = transformer_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
      torch.cuda.set_device(self.transformer.first_device)
      hidden_states = hidden_states.to(self.lm_head.weight.device)

    lm_logits = self.lm_head(hidden_states)
    mc_logits = self.multiple_choice_head(hidden_states,
                                          mc_token_ids).squeeze(-1)

    mc_loss = None
    if mc_labels is not None:
      loss_fct = CrossEntropyLoss()
      mc_loss = loss_fct(
          mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
    lm_loss = None
    if labels is not None:
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      loss_fct = CrossEntropyLoss()
      lm_loss = loss_fct(
          shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if not return_dict:
      output = (lm_logits, mc_logits) + transformer_outputs[1:]
      if mc_loss is not None:
        output = (mc_loss,) + output
      return ((lm_loss,) + output) if lm_loss is not None else output

    return GPT2DoubleHeadsModelOutput(
        loss=lm_loss,
        mc_loss=mc_loss,
        logits=lm_logits,
        mc_logits=mc_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )

  @staticmethod
  def _reorder_cache(past: Tuple[Tuple[torch.Tensor]],
                     beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
    """
        This function is used to re-order the `past_key_values` cache if
        [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match
        `past_key_values` with the correct
        beam_idx at every generation step.
        """
    return tuple(
        tuple(
            past_state.index_select(0, beam_idx.to(past_state.device))
            for past_state in layer_past)
        for layer_past in past)


@add_start_docstrings(
    """
    The GPT2 Model transformer with a sequence classification head on top (linear layer).

    [`GPT2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
  _keys_to_ignore_on_load_missing = [
      r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"
  ]

  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.transformer = GPT2Model(config)
    self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

    # Model parallel
    self.model_parallel = False
    self.device_map = None

    # Initialize weights and apply final processing
    self.post_init()

  @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
  @add_code_sample_docstrings(
      processor_class=_TOKENIZER_FOR_DOC,
      checkpoint="microsoft/DialogRPT-updown",
      output_type=SequenceClassifierOutputWithPast,
      config_class=_CONFIG_FOR_DOC,
  )
  def forward(
      self,
      input_ids=None,
      past_key_values=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    r"""

        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression
            loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed
            (Cross-Entropy).
        """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]
    logits = self.score(hidden_states)

    if input_ids is not None:
      batch_size, sequence_length = input_ids.shape[:2]
    else:
      batch_size, sequence_length = inputs_embeds.shape[:2]

    assert (self.config.pad_token_id is not None or batch_size == 1
           ), "Cannot handle batch sizes > 1 if no padding token is defined."
    if self.config.pad_token_id is None:
      sequence_lengths = -1
    else:
      if input_ids is not None:
        sequence_lengths = torch.ne(input_ids,
                                    self.config.pad_token_id).sum(-1) - 1
      else:
        sequence_lengths = -1
        logger.warning(
            f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
            f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
        )

    pooled_logits = logits[torch.arange(batch_size, device=self.device),
                           sequence_lengths]

    loss = None
    if labels is not None:
      if self.config.problem_type is None:
        if self.num_labels == 1:
          self.config.problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or
                                      labels.dtype == torch.int):
          self.config.problem_type = "single_label_classification"
        else:
          self.config.problem_type = "multi_label_classification"

      if self.config.problem_type == "regression":
        loss_fct = MSELoss()
        if self.num_labels == 1:
          loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
          loss = loss_fct(pooled_logits, labels)
      elif self.config.problem_type == "single_label_classification":
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            pooled_logits.view(-1, self.num_labels), labels.view(-1))
      elif self.config.problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(pooled_logits, labels)
    if not return_dict:
      output = (pooled_logits,) + transformer_outputs[1:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=pooled_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


@add_start_docstrings(
    """
    GPT2 Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForTokenClassification(GPT2PreTrainedModel):

  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels

    self.transformer = GPT2Model(config)
    if hasattr(config,
               "classifier_dropout") and config.classifier_dropout is not None:
      classifier_dropout = config.classifier_dropout
    elif hasattr(config,
                 "hidden_dropout") and config.hidden_dropout is not None:
      classifier_dropout = config.hidden_dropout
    else:
      classifier_dropout = 0.1
    self.dropout = nn.Dropout(classifier_dropout)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    # Model parallel
    self.model_parallel = False
    self.device_map = None

    # Initialize weights and apply final processing
    self.post_init()

  @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
  @add_code_sample_docstrings(
      processor_class=_TOKENIZER_FOR_DOC,
      checkpoint="microsoft/DialogRPT-updown",
      output_type=TokenClassifierOutput,
      config_class=_CONFIG_FOR_DOC,
  )
  def forward(
      self,
      input_ids=None,
      past_key_values=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    r"""

        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression
            loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed
            (Cross-Entropy).
        """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = transformer_outputs[0]
    hidden_states = self.dropout(hidden_states)
    logits = self.classifier(hidden_states)

    loss = None
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
      output = (logits,) + transformer_outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )
