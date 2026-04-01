# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

'''
Modification notice:
- Extracted SlotAttention class from the original "slot-attention.py" into this file 
- Replaced batchdot (from keras 2) with einsum (from keras 3)
'''

'''Slot Attention model for object discovery and set prediction.'''
import numpy as np
import tensorflow as tf
import keras.layers as layers
import keras
from keras import backend as kerasBackend

class SlotAttention(layers.Layer):
  """Slot Attention module."""

  def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size,
               epsilon=1e-8):
    """Builds the Slot Attention module.

    Args:
      num_iterations: Number of iterations.
      num_slots: Number of slots.
      slot_size: Dimensionality of slot feature vectors.
      mlp_hidden_size: Hidden layer size of MLP.
      epsilon: Offset for attention coefficients before normalization.
    """
    super().__init__()
    self.num_iterations = num_iterations
    self.num_slots = num_slots
    self.slot_size = slot_size
    self.mlp_hidden_size = mlp_hidden_size
    self.epsilon = epsilon

    self.norm_inputs = layers.LayerNormalization()
    self.norm_slots = layers.LayerNormalization()
    self.norm_mlp = layers.LayerNormalization()

    # Parameters for Gaussian init (shared by all slots).
    self.slots_mu = self.add_weight(
        initializer="glorot_uniform",
        shape=[1, 1, self.slot_size],
        dtype=tf.float32,
        name="slots_mu")
    self.slots_log_sigma = self.add_weight(
        initializer="glorot_uniform",
        shape=[1, 1, self.slot_size],
        dtype=tf.float32,
        name="slots_log_sigma")

    # Linear maps for the attention module.
    self.project_q = layers.Dense(self.slot_size, use_bias=False, name="q")
    self.project_k = layers.Dense(self.slot_size, use_bias=False, name="k")
    self.project_v = layers.Dense(self.slot_size, use_bias=False, name="v")

    # Slot update functions.
    self.gru = layers.GRUCell(self.slot_size)
    self.mlp = keras.Sequential([
        layers.Dense(self.mlp_hidden_size, activation="relu"),
        layers.Dense(self.slot_size)
    ], name="mlp")

  def call(self, inputs):
    # `inputs` has shape [batch_size, num_inputs, inputs_size].
    inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
    k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
    v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

    # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
    slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal(
        [tf.shape(inputs)[0], self.num_slots, self.slot_size]) # type: ignore

    # Multiple rounds of attention.
    for _ in range(self.num_iterations):
      slots_prev = slots
      slots = self.norm_slots(slots)

      '''
      b = batch_size
      d = slot_size
      k = num_slots
      n = num_inputs
      '''

      # Attention.
      q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
      q *= self.slot_size ** -0.5  # Normalization.
      attn_logits = keras.ops.einsum("bnd, bkd -> bnk", k, q)
      attn = tf.nn.softmax(attn_logits, axis=-1)
      # `attn` has shape: [batch_size, num_inputs, num_slots].

      # Weigted mean.
      attn += self.epsilon # type: ignore
      attn /= tf.reduce_sum(attn, axis=-2, keepdims=True)
      updates = keras.ops.einsum("bnk,bnd->bkd", attn, v)
      # `updates` has shape: [batch_size, num_slots, slot_size].

      # Slot update.
      slots, _ = self.gru(updates, [slots_prev])
      slots += self.mlp(self.norm_mlp(slots))

    return slots