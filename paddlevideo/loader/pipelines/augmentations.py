#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import paddle

from ..registry import PIPELINES
from .base import (_ARRAY, _BOX, _IMSIZE, _IMTYPE, _RESULT, _SCALE,
                   BaseOperation)


@PIPELINES.register()
class Scale(BaseOperation):
    """
    Scale images.
    Args:
        short_size(float | int): Short size of an image will be scaled to the short_size.
        fixed_ratio(bool): Set whether to zoom according to a fixed ratio. default: True
        # do_round(bool): Whether to round up when calculating the zoom ratio. default: False
        backend(str): Choose pillow or cv2 as the graphics processing backend. default: 'pillow'
    """
    def __init__(self,
                 scale_size: int,
                 keep_ratio: bool = True,
                 fixed_ratio: Union[int, float, None] = None,
                 interpolation: str = 'bilinear'):

        if keep_ratio:
            # for short side scale method
            scale_size = (np.inf, scale_size)
        else:
            scale_size = (scale_size, scale_size)
        self.scale_size = scale_size
        self.keep_ratio = keep_ratio
        if isinstance(fixed_ratio, (int, float, None)):
            if fixed_ratio <= 0:
                raise ValueError(f"fixed_ratio must > 0, but got {fixed_ratio}")
        else:
            raise TypeError(
                f"fixed_ratio must be int or float, but got {type(fixed_ratio)}"
            )
        self.fixed_ratio = fixed_ratio
        self.interpolation = interpolation
        if self.fixed_ratio and self.keep_ratio:
            raise ValueError(
                f"keep_ratio can't be true when fixed_ratio is provided")

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs resize operations.
        Args:
            imgs (Sequence[PIL.Image]): List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            resized_imgs: List where each item is a PIL.Image after scaling.
        """
        imgs = results['imgs']
        w, h = self.get_size(imgs)

        if self.fixed_ratio is not None:
            if min(w, h) == self.scale_size[1]:
                ow, oh = w, h
            elif w < h:
                ow, oh = self.scale_size[1], int(self.scale_size[1] *
                                                 self.fixed_ratio)
            else:
                ow, oh = int(self.scale_size[1] *
                             self.fixed_ratio), self.scale_size[1]
        elif self.keep_ratio:
            ow, oh = self.get_scaled_size((w, h), self.scale_size)
        else:
            ow, oh = self.scale_size

        if isinstance(imgs, paddle.Tensor):  # [*,*,h,w]
            resized_imgs = self.apply_resize(imgs, (ow, oh), self.interpolation)
        else:
            resized_imgs = [
                self.apply_resize(img, (ow, oh), self.interpolation)
                for img in imgs
            ]
        results['imgs'] = resized_imgs
        return results


@PIPELINES.register()
class RandomCrop(BaseOperation):
    """
    Random crop images.
    Args:
        target_size(int): Random crop a square with the target_size from an image.
    """
    def __init__(self, target_size: int):
        if isinstance(target_size, int):
            if target_size <= 0:
                raise ValueError(
                    f"target_size must >= 0, but got {target_size}")
        self.target_size = (target_size, target_size)

    def __call__(self, results: _RESULT) -> _RESULT:
        """
        Performs random crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            crop_imgs: List where each item is a PIL.Image after random crop.
        """
        imgs = results['imgs']
        w, h = self.get_size(imgs)

        th, tw = self.target_size, self.target_size
        if not (w >= self.target_size and h >= self.target_size):
            raise ValueError(f"The clipping edge({tw})x({th}) should\
                     not be larger than image edge({w})x({h})")

        crop_images = []
        x1 = np.random.randint(0, w - tw + 1)  # cover [0,w-tw]
        y1 = np.random.randint(0, h - th + 1)  # cover [0,h-th]
        x2 = x1 + tw
        y2 = y1 + th
        if isinstance(imgs, paddle.Tensor):  # [*,*,h,w]
            crop_images = self.apply_crop(imgs, (x1, y1, x2, y2))
        else:
            crop_images = [
                self.apply_crop(img, (x1, y1, x2, y2)) for img in imgs
            ]
        results['imgs'] = crop_images
        return results


@PIPELINES.register()
class CenterCrop(BaseOperation):
    """
    Center crop images.
    Args:
        target_size(int): Center crop a square with the target_size from an image.
        do_round(bool): Whether to round up the coordinates of the upper left corner of the cropping area. default: True
    """
    def __init__(self, target_size: int):
        if isinstance(target_size, int):
            if target_size <= 0:
                raise ValueError(
                    f"target_size must >= 0, but got {target_size}")

        self.target_size = target_size
        # self.do_round = do_round

    def __call__(self, results: _RESULT) -> _RESULT:
        """
        Performs Center crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            ccrop_imgs: List where each item is a PIL.Image after Center crop.
        """
        imgs = results['imgs']
        w, h = self.get_size(imgs)
        th, tw = self.target_size, self.target_size
        if not (w >= self.target_size and h >= self.target_size):
            raise ValueError(f"The clipping edge({tw})x({th}) should\
                     not be larger than image edge({w})x({h})")

        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        x2 = x1 + tw
        y2 = y1 + th
        if isinstance(imgs, paddle.Tensor):  # [*,*,h,w]
            crop_images = self.apply_crop(imgs, (x1, y1, x2, y2))
        else:
            crop_images = [
                self.apply_crop(img, (x1, y1, x2, y2)) for img in imgs
            ]
        results['imgs'] = crop_images
        return results


@PIPELINES.register()
class MultiScaleCrop(BaseOperation):
    """
    Random crop images in with multiscale sizes
    Args:
        target_size(int): Random crop a square with the target_size from an image.
        scales(int): List of candidate cropping scales.
        max_distort(int): Maximum allowable deformation combination distance.
        fix_crop(int): Whether to fix the cutting start point.
        allow_duplication(int): Whether to allow duplicate candidate crop starting points.
        more_fix_crop(int): Whether to allow more cutting starting points.
    """
    def __init__(
            self,
            target_size:
        int,  # NOTE: named target size now, but still pass short size in it!
            scales: int = None,
            max_distort: int = 1,
            fix_crop: bool = True,
            allow_duplication: bool = False,
            more_fix_crop: bool = True,
            interpolation: str = 'bilinear'):

        if isinstance(target_size, int):
            if target_size <= 0:
                raise ValueError(
                    f"target_size must >= 0, but got {target_size}")

        self.target_size = (target_size, target_size)
        self.scales = scales if scales else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.allow_duplication = allow_duplication
        self.more_fix_crop = more_fix_crop
        self.interpolation = interpolation

    def __call__(self, results: _RESULT) -> _RESULT:
        """
        Performs MultiScaleCrop operations.
        Args:
            imgs: List where wach item is a PIL.Image.
            XXX:
        results:

        """
        imgs = results['imgs']
        w, h = self.get_size(imgs)

        # get random crop offset
        def _sample_crop_size(im_size: _IMSIZE):
            image_w, image_h = im_size[0], im_size[1]

            base_size = min(image_w, image_h)
            crop_sizes = [int(base_size * x) for x in self.scales]
            crop_h = [
                self.target_size[1] if abs(x - self.target_size[1]) < 3 else x
                for x in crop_sizes
            ]
            crop_w = [
                self.target_size[0] if abs(x - self.target_size[0]) < 3 else x
                for x in crop_sizes
            ]

            pairs = []
            for i, h in enumerate(crop_h):
                for j, w in enumerate(crop_w):
                    if abs(i - j) <= self.max_distort:
                        pairs.append((w, h))
            crop_pair = random.choice(pairs)
            if not self.fix_crop:
                w_offset = random.randint(0, image_w - crop_pair[0])
                h_offset = random.randint(0, image_h - crop_pair[1])
            else:
                w_step = (image_w - crop_pair[0]) // 4
                h_step = (image_h - crop_pair[1]) // 4

                ret = list()
                ret.append((0, 0))  # upper left
                if self.allow_duplication or w_step != 0:
                    ret.append((4 * w_step, 0))  # upper right
                if self.allow_duplication or h_step != 0:
                    ret.append((0, 4 * h_step))  # lower left
                if self.allow_duplication or (h_step != 0 and w_step != 0):
                    ret.append((4 * w_step, 4 * h_step))  # lower right
                if self.allow_duplication or (h_step != 0 or w_step != 0):
                    ret.append((2 * w_step, 2 * h_step))  # center

                if self.more_fix_crop:
                    ret.append((0, 2 * h_step))  # center left
                    ret.append((4 * w_step, 2 * h_step))  # center right
                    ret.append((2 * w_step, 4 * h_step))  # lower center
                    ret.append((2 * w_step, 0 * h_step))  # upper center

                    ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                    ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                    ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                    ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

                w_offset, h_offset = random.choice(ret)

            return crop_pair[0], crop_pair[1], w_offset, h_offset

        tw, th, x1, y1 = _sample_crop_size((w, h))
        x2 = x1 + tw
        y2 = x2 + th
        if isinstance(imgs, paddle.Tensor):  # [*,*,h,w]
            crop_imgs = self.apply_crop(imgs, (x1, y1, x2, y2))
            resized_crop_imgs = self.apply_resize(crop_imgs, self.target_size,
                                                  self.interpolation)
        else:
            crop_imgs = [self.apply_crop(img, (x1, y1, x2, y2)) for img in imgs]
            resized_crop_imgs = [
                self.apply_resize(crop_img, (x1, y1, x2, y2))
                for crop_img in crop_imgs
            ]
        results['imgs'] = resized_crop_imgs
        return results


@PIPELINES.register()
class RandomFlip(BaseOperation):
    """
    Random Flip images.
    Args:
        p(float): Random flip images with the probability p.
    """
    def __init__(self, prob: float = 0.5, direction: str = 'horizontal'):
        self.prob = prob
        self.direction = direction

    def __call__(self, results: _RESULT) -> _RESULT:
        """
        Performs random flip operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            flip_imgs: List where each item is a PIL.Image after random flip.
        """
        imgs = results['imgs']
        flip = random.random() < self.prob
        if flip:
            if isinstance(imgs, paddle.Tensor):  # [*,*,h,w]
                imgs = self.apply_flip(imgs, direction=self.direction)
            else:
                imgs = [
                    self.apply_flip(img, direction=self.direction)
                    for img in imgs
                ]

        results['imgs'] = imgs
        return results


@PIPELINES.register()
class Image2Array(BaseOperation):
    """
    transfer PIL.Image to Numpy array and transpose dimensions from 'dhwc' to 'dchw'.
    Args:
        transpose: whether to transpose or not, default True, False for slowfast.
    """
    def __init__(self, format_shape: str = 'TCHW'):
        assert format_shape in [
            'TCHW', 'CTHW'
        ], f"Target format must in ['TCHW', 'CTHW'], but got {format_shape}"
        self.format_shape = format_shape

    def __call__(self, results: _RESULT) -> _RESULT:
        """
        Performs Image to NumpyArray operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            np_imgs: Numpy array.
        """
        imgs = results['imgs']
        if isinstance(imgs, paddle.Tensor):
            pass
        else:
            imgs = np.stack(imgs).astype('float32')
        # THWC
        if self.format_shape == 'CTHW':
            perm = (3, 0, 1, 2)
        elif self.format_shape == 'TCHW':
            perm = (0, 3, 1, 2)
        imgs = imgs.transpose(perm)
        results['imgs'] = imgs
        return results


@PIPELINES.register()
class Normalization(BaseOperation):
    """
    Normalization.
    Args:
        mean(Sequence[float]): mean values of different channels.
        std(Sequence[float]): std values of different channels.
        tensor_shape(list): size of mean, default [3,1,1]. For slowfast, [1,1,1,3]
    """
    def __init__(self,
                 mean: list,
                 std: list,
                 tensor_shape: list = [3, 1, 1],
                 to_tensor: bool = False):
        if not isinstance(mean, list):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}')
        if not isinstance(std, list):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')
        self.mean = np.array(mean).reshape(tensor_shape).astype(np.float32)
        self.std = np.array(std).reshape(tensor_shape).astype(np.float32)
        self.to_tensor = to_tensor

    def __call__(self, results: _RESULT) -> _RESULT:
        """
        Performs normalization operations.
        Args:
            imgs: Numpy array.
        return:
            np_imgs: Numpy array after normalization.
        """
        imgs = results['imgs']
        imgs = self.apply_normalization(imgs, self.mean, self.std)
        if self.to_tensor:
            imgs = paddle.to_tensor(imgs, dtype=paddle.float32, place='cpu')
        results['imgs'] = imgs
        return results


@PIPELINES.register()
class JitterScale(BaseOperation):
    """
    Scale image, while the target short size is randomly select between min_size and max_size.
    Args:
        min_size: Lower bound for random sampler.
        max_size: Higher bound for random sampler.
    """
    def __init__(self,
                 min_size,
                 max_size,
                 short_cycle_factors=[0.5, 0.7071],
                 default_min_size=256):
        self.default_min_size = default_min_size
        self.orig_min_size = self.min_size = min_size
        self.max_size = max_size
        self.short_cycle_factors = short_cycle_factors

    def __call__(self, results: _RESULT) -> _RESULT:
        """
        Performs jitter resize operations.
        Args:
            imgs (Sequence[PIL.Image]): List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            resized_imgs: List where each item is a PIL.Image after scaling.
        """
        short_cycle_idx = results.get('short_cycle_idx')
        if short_cycle_idx in [0, 1]:
            self.min_size = int(
                round(self.short_cycle_factors[short_cycle_idx] *
                      self.default_min_size))
        else:
            self.min_size = self.orig_min_size

        imgs = results['imgs']
        size = int(round(np.random.uniform(self.min_size, self.max_size)))
        assert (len(imgs) >= 1), \
            "len(imgs):{} should be larger than 1".format(len(imgs))

        w, h = self.get_size(imgs)
        if (w <= h and w == size) or (h <= w and h == size):
            return results

        ow = size
        oh = size
        if w < h:
            oh = int(math.floor((float(h) / w) * size))
        else:
            ow = int(math.floor((float(w) / h) * size))

        if isinstance(imgs, paddle.Tensor):  # [*,*,h,w]
            resized_imgs = self.apply_resize(imgs, (ow, oh), self.interpolation)
        else:
            resized_imgs = [
                self.apply_resize(img, (ow, oh), self.interpolation)
                for img in imgs
            ]
        results['imgs'] = resized_imgs
        return results


@PIPELINES.register()
class MultiCrop(BaseOperation):
    """
    Random crop image.
    This operation can perform multi-crop during multi-clip test, as in slowfast model.
    Args:
        target_size(int): Random crop a square with the target_size from an image.
    """
    def __init__(self,
                 target_size: int,
                 default_crop_size: int = 224,
                 short_cycle_factors: List[float] = [0.5, 0.7071],
                 test_mode: bool = False):
        self.orig_target_size = self.target_size = target_size
        self.short_cycle_factors = short_cycle_factors
        self.default_crop_size = default_crop_size
        self.test_mode = test_mode

    def __call__(self, results: _RESULT) -> _RESULT:
        """
        Performs random crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            crop_imgs: List where each item is a PIL.Image after random crop.
        """
        imgs = results['imgs']
        spatial_sample_index = results['spatial_sample_index']
        spatial_num_clips = results['spatial_num_clips']

        short_cycle_idx = results.get('short_cycle_idx')
        if short_cycle_idx in [0, 1]:
            self.target_size = int(
                round(self.short_cycle_factors[short_cycle_idx] *
                      self.default_crop_size))
        else:
            self.target_size = self.orig_target_size  # use saved value before call

        w, h = imgs[0].size
        if w == self.target_size and h == self.target_size:
            return results

        assert (w >= self.target_size) and (h >= self.target_size), \
            "image width({}) and height({}) should be larger than crop size({},{})".format(
                w, h, self.target_size, self.target_size)

        if not self.test_mode:
            x1 = random.randint(0, w - self.target_size)
            y1 = random.randint(0, h - self.target_size)
        else:  # multi-crop
            x_gap = int(
                math.ceil((w - self.target_size) / (spatial_num_clips - 1)))
            y_gap = int(
                math.ceil((h - self.target_size) / (spatial_num_clips - 1)))
            if h > w:
                x1 = int(math.ceil((w - self.target_size) / 2))
                if spatial_sample_index == 0:
                    y1 = 0
                elif spatial_sample_index == spatial_num_clips - 1:
                    y1 = h - self.target_size
                else:
                    y1 = y_gap * spatial_sample_index
            else:
                y1 = int(math.ceil((h - self.target_size) / 2))
                if spatial_sample_index == 0:
                    x1 = 0
                elif spatial_sample_index == spatial_num_clips - 1:
                    x1 = w - self.target_size
                else:
                    x1 = x_gap * spatial_sample_index

        x2 = x1 + self.target_size
        y2 = y1 + self.target_size
        if isinstance(imgs, paddle.Tensor):  # [*,*,h,w]
            crop_images = self.apply_crop(imgs, (x1, y1, x2, y2))
        else:
            crop_images = [
                self.apply_crop(img, (x1, y1, x2, y2)) for img in imgs
            ]
        results['imgs'] = crop_images
        return results


@PIPELINES.register()
class PackOutput(BaseOperation):
    """
    In slowfast model, we want to get slow pathway from fast pathway based on
    alpha factor.
    Args:
        alpha(int): temporal length of fast/slow
    """
    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, results: _RESULT) -> _RESULT:
        fast_pathway = results['imgs']

        # sample num points between start and end
        slow_idx_start = 0
        slow_idx_end = fast_pathway.shape[0] - 1
        slow_idx_num = fast_pathway.shape[0] // self.alpha
        slow_idxs_select = np.linspace(slow_idx_start, slow_idx_end,
                                       slow_idx_num).astype("int64")
        slow_pathway = fast_pathway[slow_idxs_select]

        # T H W C -> C T H W.
        slow_pathway = slow_pathway.transpose(3, 0, 1, 2)
        fast_pathway = fast_pathway.transpose(3, 0, 1, 2)

        # slow + fast
        frames_list = [slow_pathway, fast_pathway]
        results['imgs'] = frames_list
        return results


@PIPELINES.register()
class GroupFullResSample(BaseOperation):
    def __init__(self, crop_size: int, flip: bool = False):
        self.crop_size = (crop_size, crop_size)
        self.flip = flip

    def __call__(self, results: _RESULT) -> _RESULT:
        img_group = results['imgs']

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for x1, y1 in offsets:
            normal_group = list()
            flip_group = list()
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            for i, img in enumerate(img_group):
                crop_img = self.apply_crop(img, (x1, y1, x2, y2))
                normal_group.append(crop_img)
                if self.flip:
                    flip_crop_img = self.apply_flip(crop_img)
                    flip_group.append(flip_crop_img)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)

        results['imgs'] = oversample_group
        return results


@PIPELINES.register()
class TenCrop(BaseOperation):
    """
    Crop out 5 regions (4 corner points + 1 center point) from the picture,
    and then flip the cropping result to get 10 cropped images, which can make the prediction result more robust.
    Args:
        target_size(int | tuple[int]): (w, h) of target size for crop.
    """
    def __init__(self, target_size: int):
        self.target_size = (target_size, target_size)

    def __call__(self, results: _RESULT) -> _RESULT:
        imgs = results['imgs']
        w, h = self.get_size(imgs)
        tw, th = self.target_size
        w_step = (w - tw) // 4
        h_step = (h - th) // 4
        offsets = [
            (0, 0),
            (4 * w_step, 0),
            (0, 4 * h_step),
            (4 * w_step, 4 * h_step),
            (2 * w_step, 2 * h_step),
        ]
        img_crops = list()
        for x1, y1 in offsets:
            x2 = x1 + tw
            y2 = y1 + th
            crop_imgs = [self.apply_crop(img, (x1, y1, x2, y2)) for img in imgs]
            crop_flip_imgs = [
                self.apply_flip(crop_img) for crop_img in crop_imgs
            ]
            img_crops.extend(crop_imgs)
            img_crops.extend(crop_flip_imgs)

        results['imgs'] = img_crops
        return results


@PIPELINES.register()
class UniformCrop(BaseOperation):
    """
    Perform uniform spatial sampling on the images,
    select the two ends of the long side and the middle position (left middle right or top middle bottom) 3 regions.
    Args:
        target_size(int | tuple[int]): (w, h) of target size for crop.
    """
    def __init__(self, target_size: int):
        self.target_size = (target_size, target_size)

    def __call__(self, results: _RESULT) -> _RESULT:

        imgs = results['imgs']
        w, h = self.get_size(imgs)
        tw, th = self.target_size
        if h > w:
            offsets = [(0, 0), (0, int(math.ceil((h - th) / 2))), (0, h - th)]
        else:
            offsets = [(0, 0), (int(math.ceil((w - tw) / 2)), 0), (w - tw, 0)]
        crop_imgs_group = []
        if isinstance(imgs, paddle.Tensor):
            for x1, y1 in offsets:
                x2 = x1 + tw
                y2 = y1 + th
                crop_imgs = self.apply_crop(imgs, (x1, y1, x2, y2))
                crop_imgs_group.append(crop_imgs)  # CTHW
            crop_imgs_group = paddle.concat(crop_imgs_group, axis=1)  # C(GT)HW
        else:
            for x1, y1 in offsets:
                x2 = x1 + tw
                y2 = y1 + th
                crop_imgs = [
                    self.apply_crop(img, (x1, y1, x2, y2)) for img in imgs
                ]
                crop_imgs_group.extend(crop_imgs)
        results['imgs'] = crop_imgs_group
        return results


@PIPELINES.register()
class GroupResize(BaseOperation):
    def __init__(self,
                 height: int,
                 width: int,
                 scale: int,
                 K: List[List],
                 mode: str = 'train'):
        self.height = height
        self.width = width
        self.scale = scale
        self.resize = {}
        self.K = np.array(K, dtype=np.float32)
        self.mode = mode
        for i in range(self.scale):
            s = 2**i
            self.resize[i] = paddle.vision.transforms.Resize(
                (self.height // s, self.width // s), interpolation='lanczos')

    def __call__(self, results: _RESULT) -> _RESULT:
        if self.mode == 'infer':
            imgs: Dict[Tuple, Any] = results['imgs']
            for k in list(imgs):  # ("color", 0, -1)
                if "color" in k or "color_n" in k:
                    n, im, _ = k
                    for i in range(self.scale):
                        imgs[(n, im, i)] = self.resize[i](imgs[(n, im, i - 1)])
        else:
            imgs = results['imgs']
            for scale in range(self.scale):
                K = self.K.copy()

                K[0, :] *= self.width // (2**scale)
                K[1, :] *= self.height // (2**scale)

                inv_K = np.linalg.pinv(K)
                imgs[("K", scale)] = K
                imgs[("inv_K", scale)] = inv_K

            for k in list(imgs):
                if "color" in k or "color_n" in k:
                    n, im, i = k
                    for i in range(self.scale):
                        imgs[(n, im, i)] = self.resize[i](imgs[(n, im, i - 1)])

            results['imgs'] = imgs
        return results


@PIPELINES.register()
class ColorJitter(BaseOperation):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    """
    def __init__(self,
                 brightness: float = 0.0,
                 contrast: float = 0.0,
                 saturation: float = 0.0,
                 hue: float = 0.0,
                 mode: str = 'train',
                 prob: float = 0.5):
        self.mode = mode
        self.colorjitter = paddle.vision.transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        self.prob = prob

    def __call__(self, results: _RESULT) -> _RESULT:
        """
        Args:
            results (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """

        color_aug = self.prob < random.random()
        imgs = results['imgs']
        for k in list(imgs):
            f = imgs[k]
            if "color" in k or "color_n" in k:
                n, im, i = k
                imgs[(n, im, i)] = f
                if color_aug:
                    imgs[(n + "_aug", im, i)] = self.colorjitter(f)
                else:
                    imgs[(n + "_aug", im, i)] = f
        if self.mode == "train":
            for i in results['frame_idxs']:
                del imgs[("color", i, -1)]
                del imgs[("color_aug", i, -1)]
                del imgs[("color_n", i, -1)]
                del imgs[("color_n_aug", i, -1)]
        else:
            for i in results['frame_idxs']:
                del imgs[("color", i, -1)]
                del imgs[("color_aug", i, -1)]

        results['img'] = imgs
        return results


@PIPELINES.register()
class GroupRandomFlip(BaseOperation):
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, results: _RESULT) -> _RESULT:

        imgs = results['imgs']
        do_flip = self.prob < random.random()
        if do_flip:
            for k in list(imgs):
                if "color" in k or "color_n" in k:
                    n, im, i = k
                    imgs[(n, im, i)] = self.apply_flip(imgs[(n, im, i)], )
            if "depth_gt" in imgs:
                imgs['depth_gt'] = np.array(np.fliplr(imgs['depth_gt']))

        results['imgs'] = imgs
        return results


@PIPELINES.register()
class ToArray(BaseOperation):
    def __init__(self):
        pass

    def __call__(self, results: _RESULT):
        imgs = results['imgs']
        for k in list(imgs):
            if "color" in k or "color_n" in k or "color_aug" in k or "color_n_aug" in k:
                n, im, i = k
                imgs[(n, im,
                      i)] = np.array(imgs[(n, im, i)]).astype('float32') / 255.0
                imgs[(n, im, i)] = imgs[(n, im, i)].transpose((2, 0, 1))
        if "depth_gt" in imgs:
            imgs['depth_gt'] = np.array(imgs['depth_gt']).astype('float32')

        results['imgs'] = imgs
        return results
