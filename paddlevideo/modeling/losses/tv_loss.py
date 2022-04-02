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

import paddle

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register()
class TVLoss(BaseWeightedLoss):
    def __init__(self):
        super(TVLoss, self).__init__()

    def _forward(self, x: paddle.Tensor) -> paddle.Tensor:
        h_x = x.shape[2]
        w_x = x.shape[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = paddle.pow((x[:, :, 1:, :]) - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = paddle.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return h_tv / count_h + w_tv / count_w

    def _tensor_size(self, t: paddle.Tensor) -> int:
        return t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3]
