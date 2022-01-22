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
from typing import Any, Dict, Sequence, Tuple, Union
from paddle.vision import transforms
import numpy as np
import paddle

from ..registry import PIPELINES
from .base import _ARRAY, _BOX, _IMSIZE, _RESULT, BaseOperation


@PIPELINES.register()
class Scale(BaseOperation):
    """Scale images

    Args:
        scale_size (int): Short size of an image, which will be scaled to scale_size
        keep_ratio (bool, optional): Whether keep original edge ratio. Defaults to True.
        fixed_ratio (Union[int, float, False], optional): Whether use fixed ratio instead of original edge ratio. Defaults to False.
        interpolation (str, optional): Interpolation method. Defaults to 'bilinear'.

    """
    def __init__(self,
                 scale_size: int,
                 keep_ratio: bool = True,
                 fixed_ratio: Union[int, float, bool] = False,
                 interpolation: str = 'bilinear'):
        if keep_ratio:
            scale_size: Tuple[float,
                              int] = (np.inf, scale_size)  # short side scale
            if fixed_ratio is not False:
                raise ValueError(
                    f"fixed_ratio must be False when keep_ratio is True")
        else:
            if fixed_ratio is False:
                scale_size: Tuple[int,
                                  int] = (scale_size, scale_size
                                          )  # scale both side to scale_size
            else:
                fixed_ratio = eval(fixed_ratio)  # convert str to float
                if not isinstance(fixed_ratio, (int, float)):
                    raise ValueError(
                        f"fixed ratio must be int or float or False, but got {type(fixed_ratio)}"
                    )
        self.scale_size = scale_size
        self.keep_ratio = keep_ratio
        self.fixed_ratio = fixed_ratio
        self.interpolation = interpolation
        if self.fixed_ratio is not False and self.keep_ratio:
            raise ValueError(
                f"keep_ratio can't be true when fixed_ratio is provided")

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply scale operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        w, h = self.get_im_size(imgs)

        if self.fixed_ratio is not False:
            if min(w, h) == self.scale_size:
                ow, oh = w, h
            elif w < h:
                ow, oh = self.scale_size, int(self.scale_size *
                                              self.fixed_ratio)
            else:
                ow, oh = int(self.scale_size *
                             self.fixed_ratio), self.scale_size
        elif self.keep_ratio:
            ow, oh = self.get_scaled_size((w, h), self.scale_size)
        else:
            ow, oh = self.scale_size

        if self.isTensor(imgs):  # [*,*,h,w]
            resized_imgs = self.im_resize(imgs, (ow, oh), self.interpolation)
        else:
            resized_imgs = [
                self.im_resize(img, (ow, oh), self.interpolation)
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
        self.target_size = (target_size, target_size)

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply randomcrop operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        w, h = self.get_im_size(imgs)

        th, tw = self.target_size
        if not (w >= self.target_size[1] and h >= self.target_size[0]):
            raise ValueError(f"The clipping edge({tw})x({th}) should\
                     not be larger than image edge({w})x({h})")

        crop_images = []
        x1 = np.random.randint(0, w - tw + 1)  # cover [0,w-tw]
        y1 = np.random.randint(0, h - th + 1)  # cover [0,h-th]
        x2 = x1 + tw
        y2 = y1 + th
        if self.isTensor(imgs):  # [*,*,h,w]
            crop_images = self.im_crop(imgs, (x1, y1, x2, y2))
        else:
            crop_images = [self.im_crop(img, (x1, y1, x2, y2)) for img in imgs]
        results['imgs'] = crop_images
        return results


@PIPELINES.register()
class RandomResizedCrop(RandomCrop):
    """A crop of random size (default: of 0.08 to 1.0) of the original size and a random
       aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made.

    Args:
        area_range (Tuple[float, float], optional): [description]. Defaults to (0.08, 1.0).
        aspect_ratio_range (Tuple[float, float], optional): [description]. Defaults to (3 / 4, 4 / 3).
    """
    def __init__(self,
                 area_range: Tuple[float, float] = (0.08, 1.0),
                 aspect_ratio_range: Tuple[float, float] = (3 / 4, 4 / 3)):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range

    @staticmethod
    def get_crop_bbox(img_shape: _IMSIZE,
                      area_range: Tuple[float, float],
                      aspect_ratio_range: Tuple[float, float],
                      max_attempts: int = 10) -> _BOX:

        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(np.log(min_ar), np.log(max_ar),
                              size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results: _RESULT) -> _RESULT:
        imgs = results['imgs']
        img_w, img_h = self.get_im_size(imgs)

        x1, y1, x2, y2 = self.get_crop_bbox((img_h, img_w), self.area_range,
                                            self.aspect_ratio_range)

        if self.isTensor(imgs):
            imgs = self.im_crop(imgs, (x1, y1, x2, y2))
        else:
            imgs = [self.im_crop(img, (x1, y1, x2, y2)) for img in imgs]
        results['imgs'] = imgs
        return results


@PIPELINES.register()
class CenterCrop(BaseOperation):
    """Center crop images.

    Args:
        target_size (int): Center crop a square with the target_size from an image.
    """
    def __init__(self, target_size: int):
        self.target_size = (target_size, target_size)

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply centercrop operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        w, h = self.get_im_size(imgs)
        th, tw = self.target_size
        if not (w >= self.target_size[1] and h >= self.target_size[0]):
            raise ValueError(f"The clipping edge({tw})x({th}) should\
                     not be larger than image edge({w})x({h})")

        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        x2 = x1 + tw
        y2 = y1 + th
        if self.isTensor(imgs):  # [*,*,h,w]
            crop_images = self.im_crop(imgs, (x1, y1, x2, y2))
        else:
            crop_images = [self.im_crop(img, (x1, y1, x2, y2)) for img in imgs]
        results['imgs'] = crop_images
        return results


@PIPELINES.register()
class MultiScaleCrop(BaseOperation):
    """Random crop images in with multiscale sizes

    Args:
        target_size (int): Random crop a square with the target_size from an image.
        scales (int, optional): List of candidate cropping scales.. Defaults to None.
        max_distort (int, optional): Maximum allowable deformation combination distance. Defaults to 1.
        fix_crop (bool, optional): Whether to fix the cutting start point. Defaults to True.
        allow_duplication (bool, optional): Whether to allow duplicate candidate crop starting points. Defaults to False.
        more_fix_crop (bool, optional): Whether to allow more cutting starting points. Defaults to True.
        interpolation (str, optional): Interpolation method. Defaults to 'bilinear'.
    """
    def __init__(self,
                 target_size: int,
                 scales: _ARRAY = None,
                 max_distort: int = 1,
                 fix_crop: bool = True,
                 allow_duplication: bool = False,
                 more_fix_crop: bool = True,
                 interpolation: str = 'bilinear'):
        self.target_size = (target_size, target_size)
        self.scales = scales if scales else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.allow_duplication = allow_duplication
        self.more_fix_crop = more_fix_crop
        self.interpolation = interpolation

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply multiscalecrop operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        w, h = self.get_im_size(imgs)

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
        y2 = y1 + th
        if self.isTensor(imgs):  # [*,*,h,w]
            crop_imgs = self.im_crop(imgs, (x1, y1, x2, y2))
            resized_crop_imgs = self.im_resize(crop_imgs, self.target_size,
                                               self.interpolation)
        else:
            crop_imgs = [self.im_crop(img, (x1, y1, x2, y2)) for img in imgs]
            resized_crop_imgs = [
                self.im_resize(crop_img, self.target_size, self.interpolation)
                for crop_img in crop_imgs
            ]
        results['imgs'] = resized_crop_imgs
        return results


@PIPELINES.register()
class RandomFlip(BaseOperation):
    """Random Flip images.

    Args:
        prob (float, optional): Random flip images with the probability prob. Defaults to 0.5.
        direction (str, optional): Flip direction. Defaults to 'horizontal'.
    """
    def __init__(self, prob: float = 0.5, direction: str = 'horizontal'):
        self.prob = prob
        self.direction = direction

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply randomflip operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        do_flip = random.random() < self.prob
        if do_flip:
            if self.isTensor(imgs):  # [*,*,h,w]
                imgs = self.im_flip(imgs, direction=self.direction)
            else:
                imgs = [
                    self.im_flip(img, direction=self.direction) for img in imgs
                ]

        results['imgs'] = imgs
        return results


@PIPELINES.register()
class Image2Array(BaseOperation):
    """Transfer image(s) list to arrays like numpy.ndarray with certain format.

    Args:
        format_shape (str, optional): format shape. Defaults to 'TCHW'.
    """
    def __init__(self, format_shape: str = 'TCHW'):
        assert format_shape in [
            'TCHW', 'CTHW', 'THWC'
        ], f"Target format must in ['TCHW', 'CTHW', 'THWC'], but got {format_shape}"
        self.format_shape = format_shape

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply image2array operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        if isinstance(imgs, (list, tuple)):
            imgs = self.im_stack(imgs, axis=0)
            imgs = imgs.astype('float32')
        # imgs's shape is THWC now

        # transpose to target shape permutation
        if self.format_shape == 'CTHW':
            perm = (3, 0, 1, 2)
        elif self.format_shape == 'TCHW':
            perm = (0, 3, 1, 2)
        elif self.format_shape == 'THWC':
            perm = (0, 1, 2, 3)
        else:
            raise ValueError(
                f"format shape only support 'CTHW' and 'TCHW', but got {self.format_shape}"
            )
        if perm != (0, 1, 2, 3):
            imgs = imgs.transpose(perm)
        results['imgs'] = imgs
        return results


@PIPELINES.register()
class Normalization(BaseOperation):
    """Normalization

    Args:
        mean (list): mean values of different channels.
        std (list): std values of different channels.
        scale_factor (Union[int, float, None], optional): Whether scale pixel values by a factor before normalization. Defaults to 255.
        tensor_shape (Sequence[int], optional): size of mean. Defaults to [3, 1, 1].
        to_tensor (bool, optional): Whether convert normalization result to tensor. Defaults to False.
        inplace (bool, optional): Whether do normalizations inplace(if available) . Defaults to False.
    """
    def __init__(self,
                 mean: _ARRAY,
                 std: _ARRAY,
                 scale_factor: Union[int, float] = 255,
                 tensor_shape: Sequence[int] = [3, 1, 1],
                 to_tensor: bool = False,
                 inplace: bool = False):
        if not isinstance(mean, list):
            raise TypeError(f'mean must be list, but got {type(mean)}')
        if not isinstance(std, list):
            raise TypeError(f'std must be list, but got {type(std)}')
        if not isinstance(scale_factor, (int, float)):
            raise TypeError(
                f'scale factor must be int or float or None, but got {type(scale_factor)}'
            )
        self.mean = np.array(mean).reshape(tensor_shape).astype(np.float32)
        self.std = np.array(std).reshape(tensor_shape).astype(np.float32)
        self.scale_factor = scale_factor
        self.to_tensor = to_tensor
        self.inplace = inplace

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply normalization operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        if self.scale_factor != 1:
            norm_imgs = [img / self.scale_factor for img in imgs]

        norm_imgs = [
            self.im_norm(img, self.mean, self.std, self.inplace) for img in imgs
        ]
        if self.to_tensor:
            norm_imgs = self.im_stack(norm_imgs, axis=0)
            norm_imgs = paddle.to_tensor(norm_imgs, stop_gradient=True)

        results['imgs'] = norm_imgs
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
                 min_size: int,
                 max_size: int,
                 short_cycle_factors: _ARRAY = [0.5, 0.7071],
                 default_min_size: int = 256,
                 interpolation: str = 'bilinear'):
        self.min_size = min_size
        self.max_size = max_size
        self.short_cycle_factors = short_cycle_factors
        self.default_min_size = default_min_size
        self.interpolation = interpolation
        self.orig_min_size = min_size

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply jitterscale operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        short_cycle_idx = results.get('short_cycle_idx')
        if short_cycle_idx in [0, 1]:
            self.min_size = int(
                round(self.short_cycle_factors[short_cycle_idx] *
                      self.default_min_size))
        else:
            self.min_size = self.orig_min_size

        imgs = results['imgs']
        random_short_size = int(
            round(np.random.uniform(self.min_size, self.max_size)))
        assert (len(imgs) >= 1), \
            "len(imgs):{} should be larger than 1".format(len(imgs))

        w, h = self.get_im_size(imgs)
        if (w <= h and w == random_short_size) or (h <= w
                                                   and h == random_short_size):
            return results

        if w < h:
            ow = random_short_size
            oh = int(math.floor((float(h) / w) * ow))
        else:
            oh = random_short_size
            ow = int(math.floor((float(w) / h) * oh))

        if self.isTensor(imgs):  # [*,*,h,w]
            resized_imgs = self.im_resize(imgs, (ow, oh), self.interpolation)
        else:
            resized_imgs = [
                self.im_resize(img, (ow, oh), self.interpolation)
                for img in imgs
            ]
        results['imgs'] = resized_imgs
        return results


@PIPELINES.register()
class MultiCrop(BaseOperation):
    """Random crop image.

    Args:
        target_size (int): Random crop a square with the target_size from an image
        default_crop_size (int, optional): default_crop_size. Defaults to 224.
        short_cycle_factors (List[float], optional): short_cycle_factors. Defaults to [0.5, 0.7071].
        test_mode (bool, optional): Whether in test mode. Defaults to False.
    """
    def __init__(self,
                 target_size: int,
                 default_crop_size: int = 224,
                 short_cycle_factors: _ARRAY = [0.5, 0.7071],
                 test_mode: bool = False):
        self.orig_target_size = self.target_size = target_size
        self.short_cycle_factors = short_cycle_factors
        self.default_crop_size = default_crop_size
        self.test_mode = test_mode

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply multicrop operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
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
        if self.isTensor(imgs):  # [*,*,h,w]
            crop_images = self.im_crop(imgs, (x1, y1, x2, y2))
        else:
            crop_images = [self.im_crop(img, (x1, y1, x2, y2)) for img in imgs]
        results['imgs'] = crop_images
        return results


@PIPELINES.register()
class PackOutput(BaseOperation):
    """In slowfast model, we want to get slow pathway from fast pathway based on alpha factor.

    Args:
        alpha (int): temporal length of fast/slow
    """
    def __init__(self, alpha: int):
        self.alpha = alpha

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply packoutput operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
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
    """GroupFullResSample

    Args:
        crop_size (int): crop size
        flip (bool, optional): Whether take extra flip. Defaults to False.
    """
    def __init__(self, crop_size: int, flip: bool = False):
        self.crop_size = (crop_size, crop_size)
        self.flip = flip

    def __call__(self, results: _RESULT) -> _RESULT:
        img_group = results['imgs']

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = []
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = []
        for x1, y1 in offsets:
            crop_group = []
            flip_crop_group = []
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            for i, img in enumerate(img_group):
                crop_img = self.im_crop(img, (x1, y1, x2, y2))
                crop_group.append(crop_img)
                if self.flip:
                    flip_crop_img = self.im_flip(crop_img)
                    flip_crop_group.append(flip_crop_img)

            oversample_group.extend(crop_group)
            if self.flip:
                oversample_group.extend(flip_crop_group)

        results['imgs'] = oversample_group
        return results


@PIPELINES.register()
class TenCrop(BaseOperation):
    """Crop out 5 regions (4 corner points + 1 center point) from the picture,
        and then flip the cropping result to get 10 cropped images, which can make the prediction result more robust.

    Args:
        target_size (int): (target_size, target_size) of target size for crop
    """
    def __init__(self, target_size: int):
        self.target_size = (target_size, target_size)

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply groupfullressample operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        w, h = self.get_im_size(imgs)
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
            crop_imgs = [self.im_crop(img, (x1, y1, x2, y2)) for img in imgs]
            crop_flip_imgs = [self.im_flip(crop_img) for crop_img in crop_imgs]
            img_crops.extend(crop_imgs)
            img_crops.extend(crop_flip_imgs)

        results['imgs'] = img_crops
        return results


@PIPELINES.register()
class UniformCrop(BaseOperation):
    """Perform uniform spatial sampling on the images,
        select the two ends of the long side and the middle position (left middle right or top middle bottom) 3 regions.

    Args:
        target_size (int): (target_size, target_size) of target size for crop.
    """
    def __init__(self, target_size: int):
        self.target_size = (target_size, target_size)

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply uniformcrop operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        w, h = self.get_im_size(imgs)
        tw, th = self.target_size
        if h > w:
            offsets = [(0, 0), (0, int(math.ceil((h - th) / 2))), (0, h - th)]
        else:
            offsets = [(0, 0), (int(math.ceil((w - tw) / 2)), 0), (w - tw, 0)]
        crop_imgs_group = []
        if self.isTensor(imgs):
            for x1, y1 in offsets:
                x2 = x1 + tw
                y2 = y1 + th
                crop_imgs = self.im_crop(imgs, (x1, y1, x2, y2))
                crop_imgs_group.append(crop_imgs)  # CTHW
            crop_imgs_group = paddle.concat(crop_imgs_group,
                                            axis=1)  # C(T+T+T)HW
        else:
            for x1, y1 in offsets:
                x2 = x1 + tw
                y2 = y1 + th
                crop_imgs = [
                    self.im_crop(img, (x1, y1, x2, y2)) for img in imgs
                ]
                crop_imgs_group.extend(crop_imgs)
        results['imgs'] = crop_imgs_group
        return results


@PIPELINES.register()
class GroupResize(BaseOperation):
    """Resize images in image pyramid

    Args:
        height (int): image height
        width (int): image width
        scale (int): number of scales in image pyramid
        K (List[List]): Camera intrinsics matrix.
        mode (str, optional): [description]. Defaults to 'train'.
        interpolation (str, optional): Interpolation method. Defaults to 'lanczos'.
    """
    def __init__(self,
                 height: int,
                 width: int,
                 scale: int,
                 K: _ARRAY,
                 mode: str = 'train',
                 interpolation: str = 'lanczos'):
        self.height = height
        self.width = width
        self.scale = scale
        self.resize = {}
        self.K = np.array(K, dtype=np.float32)
        self.mode = mode
        for i in range(self.scale):
            s = 2**i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s),
                interpolation=interpolation)

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply groupresize operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs: Dict[Union[Tuple[str, int, int], Tuple[str, int]],
                   Any] = results['imgs']
        if self.mode == 'infer':
            for k in list(imgs):  # ("color", 0, -1)
                if "color" in k or "color_n" in k:
                    n, im, _ = k
                    for i in range(self.scale):
                        imgs[(n, im, i)] = self.resize[i](imgs[(n, im, i - 1)])
        else:
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

    Args:
        brightness (float, optional): brightness. Defaults to 0.0.
        contrast (float, optional): contrast. Defaults to 0.0.
        saturation (float, optional): saturation. Defaults to 0.0.
        hue (float, optional): hue. Defaults to 0.0.
        test_mode (bool, optional): test_mode. Defaults to 'train'.
        prob (float, optional): Whether do colorjitter with probability prob. Defaults to 0.5.
    """
    def __init__(self,
                 brightness: float = 0.0,
                 contrast: float = 0.0,
                 saturation: float = 0.0,
                 hue: float = 0.0,
                 test_mode: bool = False,
                 prob: float = 0.5):
        self.test_mode = test_mode
        self.colorjitter = transforms.ColorJitter(brightness, contrast,
                                                  saturation, hue)
        self.prob = prob

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply colorjitter operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        do_color_aug = self.prob < random.random()
        for k in list(imgs):
            f = imgs[k]
            if "color" in k or "color_n" in k:
                n, im, i = k
                imgs[(n, im, i)] = f
                if do_color_aug:
                    imgs[(n + "_aug", im, i)] = self.colorjitter(f)
                else:
                    imgs[(n + "_aug", im, i)] = f
        if self.test_mode:
            for i in results['frame_idxs']:
                del imgs[("color", i, -1)]
                del imgs[("color_aug", i, -1)]
        else:
            for i in results['frame_idxs']:
                del imgs[("color", i, -1)]
                del imgs[("color_aug", i, -1)]
                del imgs[("color_n", i, -1)]
                del imgs[("color_n_aug", i, -1)]

        results['img'] = imgs
        return results


@PIPELINES.register()
class GroupRandomFlip(BaseOperation):
    """GroupRandomFlip

    Args:
        prob (float, optional): Whether do flip with probability prob. Defaults to 0.5.
        direction (str, optional): Flip direction. Defaults to 'horizontal'.
    """
    def __init__(self, prob: float = 0.5, direction: str = 'horizontal'):
        self.prob = prob
        self.direction = direction

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply grouprandomflip operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
        imgs = results['imgs']
        do_flip = self.prob < random.random()
        if do_flip:
            for k in list(imgs):
                if "color" in k or "color_n" in k:
                    n, im, i = k
                    imgs[(n, im, i)] = self.im_flip(imgs[(n, im, i)],
                                                    self.direction)
            if "depth_gt" in imgs:
                imgs['depth_gt'] = self.im_flip(imgs['depth_gt'])

        results['imgs'] = imgs
        return results


@PIPELINES.register()
class ToArray(BaseOperation):
    """Convert images to array.
    """
    def __init__(self):
        pass

    def __call__(self, results: _RESULT) -> _RESULT:
        """Apply toarray operations on images

        Args:
            results (Dict[str, Any]): Data processed on the pipeline which is as input for the next operation.

        Returns:
            Dict[str, Any]: Processed data.
        """
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
