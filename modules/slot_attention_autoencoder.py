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
- Split implementation of SlotAttentionAutoEncoder  from slot-attention.py up into multiple files, 
  combining them in this one
'''

import keras.layers as layers
import keras
import tensorflow as tf
from rgbd_backbones.mobilenet import get_mobilenet_backbone
from spatial_broadcast_decoder import SpatialBroadcastDecoder
from soft_position_embed import SoftPositionEmbed
from slot_attention import SlotAttention
import utils

class SlotAttentionAutoEncoder(layers.Layer):
    def __init__(self, resolution: tuple[int, int], num_slots: int, num_iterations: int):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = get_mobilenet_backbone(resolution)
        self.encoder_pos = SoftPositionEmbed(64, resolution)
        self.decoder = SpatialBroadcastDecoder()

        self.layer_norm = layers.LayerNormalization()
        self.mlp = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64)
        ], name="feedforward")

        self.slot_attention = SlotAttention(
            num_iterations=num_iterations,
            num_slots=num_slots,
            slot_size=64,
            mlp_hidden_size=128
        )

    def call(self, image):
        # `image` has shape: [batch_size, width, height, num_channels].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = self.encoder_pos(x)  # Position embedding.
        x = utils.spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
        x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        return self.decoder(slots)