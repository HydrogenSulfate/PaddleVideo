# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
from paddle.nn import Linear

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_, trunc_normal_


@HEADS.register()
class ppTimeSformerHead(BaseHead):
    """TimeSformerHead Head.

    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channles in input feature.
        loss_cfg (dict): Config for building config. Default: dict(name='CrossEntropyLoss').
        std(float): Std(Scale) value in normal initilizar. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to initialize.

    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cfg=dict(name='CrossEntropyLoss'),
                 std=0.02,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cfg, **kwargs)
        self.std = std
        self.fc_cls = Linear(self.in_channels, self.num_classes)
        self.fc_dist = Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the FC layer parameters"""

        weight_init_(self.fc_cls,
                     'TruncatedNormal',
                     'fc_0.w_0',
                     'fc_0.b_0',
                     mean=0.0,
                     std=self.std)
        weight_init_(self.fc_dist,
                     'TruncatedNormal',
                     'fc_0.w_0',
                     'fc_0.b_0',
                     mean=0.0,
                     std=self.std)
        # NOTE: Temporarily use trunc_normal_ instead of TruncatedNormal
        trunc_normal_(self.fc_cls.weight, std=self.std)
        trunc_normal_(self.fc_dist.weight, std=self.std)

    def forward(self, x):
        """Define how the head is going to run.
        Args:
            x (paddle.Tensor): The input data.
        Returns:
            score: (paddle.Tensor) The classification scores for input samples.
        """
        # XXX: check dropout location!
        # x.shape = [N, embed_dim]
        x, x_dist = x
        score = self.fc_cls(x)
        score_dist = self.fc_dist(x_dist)
        # [N, num_class]
        # x = F.softmax(x)  # NOTE remove
        if self.training:
            return score, score_dist
        else:
            return (score + score_dist) / 2
