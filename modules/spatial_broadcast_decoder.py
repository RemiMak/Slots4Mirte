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
- Combined 
    Extracted the spatial_broadcast decoder from "SlotAttentionAutoEncoder" class in the original 
    "slot-attention.py" into this file
'''

import numpy as np
import keras
from keras import layers
import tensorflow as tf
from .soft_position_embed import SoftPositionEmbed

def spatial_broadcast(slots, resolution):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  slots = tf.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
  grid = tf.tile(slots, [1, resolution[0], resolution[1], 1])
  # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid

def unstack_and_split(x, batch_size, num_channels=3):
  """Unstack batch dimension and split into channels and alpha mask."""
  unstacked = tf.reshape(x, [batch_size, -1] + x.shape.as_list()[1:])
  channels, masks = tf.split(unstacked, [num_channels, 1], axis=-1)
  return channels, masks

class SpatialBroadcastDecoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.decoder_initial_size = (8, 8)
        self.decoder_cnn = keras.Sequential([
            layers.Conv2DTranspose(
                64, 5, strides=(2, 2), padding="SAME", activation="relu"),
            layers.Conv2DTranspose(
                64, 5, strides=(2, 2), padding="SAME", activation="relu"),
            layers.Conv2DTranspose(
                64, 5, strides=(2, 2), padding="SAME", activation="relu"),
            layers.Conv2DTranspose(
                64, 5, strides=(2, 2), padding="SAME", activation="relu"),
            layers.Conv2DTranspose(
                64, 5, strides=(1, 1), padding="SAME", activation="relu"),
            layers.Conv2DTranspose(
                4, 3, strides=(1, 1), padding="SAME", activation=None)
        ], name="decoder_cnn")

        self.decoder_pos = SoftPositionEmbed(64, self.decoder_initial_size)
    
    def call(self, resolution: tuple[int, int], slots: tuple[int, int, int]):
        # `slots` has shape: [batch_size, num_slots, slot_size]

        x = spatial_broadcast(slots, self.decoder_initial_size)
        # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].
       
       # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = unstack_and_split(x, batch_size=resolution[0])
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = tf.nn.softmax(masks, axis=1)
        recon_combined = tf.reduce_sum(recons * masks, axis=1)  # Recombine image.
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots