#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from typing import Optional, Tuple, Union, Sequence

import cv2
import numpy as np
from PIL import Image

from ..registry import PIPELINES
from .base import (_ARRAY, _BOX, _IMSIZE, _IMTYPE, _RESULT, CV2_INTERP_CODES,
                   PILLOW_INTERP_CODES, BaseOperation)


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


def imresize(
    img: _IMTYPE,
    size: _IMSIZE,
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    out=None,
    backend: Optional[str] = None
) -> Union[_IMTYPE, Tuple[_IMTYPE, float, float]]:
    """Resize image to a given size.  """
    h, w = img.shape[:2]
    if backend is None:
        backend = 'cv2'
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, PILLOW_INTERP_CODES[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(img,
                                 size,
                                 dst=out,
                                 interpolation=CV2_INTERP_CODES[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


@PIPELINES.register()
class EntityBoxRescale(BaseOperation):
    """Rescale the entity box and proposals according to the image shape.

    Required keys are "proposals", "gt_bboxes", added or modified keys are
    "gt_bboxes". If original "proposals" is not None, "proposals" and
    will be added or modified.

    Args:
        scale_factor (np.ndarray): The scale factor used entity_box rescaling.
    """
    def __init__(self, scale_factor: np.ndarray) -> None:
        self.scale_factor = scale_factor

    def __call__(self, results: _RESULT) -> _RESULT:
        scale_factor = np.concatenate([self.scale_factor, self.scale_factor])

        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            results['gt_bboxes'] = gt_bboxes * scale_factor

        if 'proposals' in results:
            proposals = results['proposals']
            if proposals is not None:
                assert proposals.shape[1] == 4, (
                    'proposals shape should be in '
                    f'(n, 4), but got {proposals.shape}')
                results['proposals'] = proposals * scale_factor

        return results


@PIPELINES.register()
class EntityBoxCrop(BaseOperation):
    """Crop the entity boxes and proposals according to the cropped images.

    Required keys are "proposals", "gt_bboxes", added or modified keys are
    "gt_bboxes". If original "proposals" is not None, "proposals" will be
    modified.

    Args:
        crop_bbox(np.ndarray | None): The bbox used to crop the original image.
    """
    def __init__(self, crop_bbox: Optional[_BOX]) -> None:
        self.crop_bbox = crop_bbox

    def __call__(self, results: _RESULT) -> _RESULT:
        proposals = results['proposals']
        gt_bboxes = results['gt_bboxes']

        if self.crop_bbox is None:
            return results

        x1, y1, x2, y2 = self.crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        assert gt_bboxes.shape[-1] == 4
        gt_bboxes_ = gt_bboxes.copy()
        gt_bboxes_[..., 0::2] = np.clip(gt_bboxes[..., 0::2] - x1, 0, img_w - 1)
        gt_bboxes_[..., 1::2] = np.clip(gt_bboxes[..., 1::2] - y1, 0, img_h - 1)
        results['gt_bboxes'] = gt_bboxes_

        if proposals is not None:
            assert proposals.shape[-1] == 4
            proposals_ = proposals.copy()
            proposals_[..., 0::2] = np.clip(proposals[..., 0::2] - x1, 0,
                                            img_w - 1)
            proposals_[..., 1::2] = np.clip(proposals[..., 1::2] - y1, 0,
                                            img_h - 1)
            results['proposals'] = proposals_
        return results


@PIPELINES.register()
class EntityBoxFlip(BaseOperation):
    """Flip the entity boxes and proposals with a probability.

    Reverse the order of elements in the given bounding boxes and proposals
    with a specific direction. The shape of them are preserved, but the
    elements are reordered. Only the horizontal flip is supported (seems
    vertical flipping makes no sense). Required keys are "proposals",
    "gt_bboxes", added or modified keys are "gt_bboxes". If "proposals"
    is not None, it will also be modified.

    Args:
        img_shape (tuple[int]): The img shape.
    """
    def __init__(self, img_shape: _IMSIZE) -> None:
        self.img_shape = img_shape

    def __call__(self, results: _RESULT) -> _RESULT:
        proposals = results['proposals']
        gt_bboxes = results['gt_bboxes']
        img_h, img_w = self.img_shape

        assert gt_bboxes.shape[-1] == 4
        gt_bboxes_ = gt_bboxes.copy()
        gt_bboxes_[..., 0::4] = img_w - gt_bboxes[..., 2::4] - 1
        gt_bboxes_[..., 2::4] = img_w - gt_bboxes[..., 0::4] - 1
        if proposals is not None:
            assert proposals.shape[-1] == 4
            proposals_ = proposals.copy()
            proposals_[..., 0::4] = img_w - proposals[..., 2::4] - 1
            proposals_[..., 2::4] = img_w - proposals[..., 0::4] - 1
        else:
            proposals_ = None

        results['proposals'] = proposals_
        results['gt_bboxes'] = gt_bboxes_

        return results


@PIPELINES.register()
class Resize(BaseOperation):
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "resize_size". Required keys in "lazy" is None, added or modified key is
    "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    def __init__(self,
                 scale: Union[float, _IMSIZE],
                 keep_ratio: bool = True,
                 interpolation: str = 'bilinear',
                 lazy: bool = False) -> None:
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def __call__(self, results: _RESULT) -> _RESULT:
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = self.get_scaled_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)
        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            results['imgs'] = [
                imresize(img, (new_w, new_h), interpolation=self.interpolation)
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        # if 'gt_bboxes' in results:
        assert not self.lazy
        entity_box_rescale = EntityBoxRescale(self.scale_factor)
        results = entity_box_rescale(results)

        return results


@PIPELINES.register()
class RandomRescale(BaseOperation):
    """Randomly resize images so that the short_edge is resized to a specific
    size in a given range. The scale ratio is unchanged after resizing.
    """
    def __init__(self,
                 scale_range: Union[str, Tuple],
                 interpolation: str = 'bilinear') -> None:
        scale_range = eval(scale_range)
        self.scale_range = scale_range

        assert len(scale_range) == 2
        assert scale_range[0] < scale_range[1]
        assert np.all([x > 0 for x in scale_range])

        self.keep_ratio = True
        self.interpolation = interpolation

    def __call__(self, results: _RESULT) -> _RESULT:
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        short_edge = np.random.randint(self.scale_range[0],
                                       self.scale_range[1] + 1)
        resize = Resize((-1, short_edge),
                        keep_ratio=True,
                        interpolation=self.interpolation,
                        lazy=False)
        results = resize(results)

        results['short_edge'] = short_edge
        return results


@PIPELINES.register()
class Rescale(BaseOperation):
    """resize images so that the short_edge is resized to a specific
    size in a given range. The scale ratio is unchanged after resizing.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "resize_size",
    "short_edge".

    Args:
        scale_range (tuple[int]): The range of short edge length. A closed
            interval.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """
    def __init__(self, scale_range, interpolation: str = 'bilinear') -> None:
        scale_range = eval(scale_range)
        self.scale_range = scale_range

        self.keep_ratio = True
        self.interpolation = interpolation

    def __call__(self, results: _RESULT) -> _RESULT:
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        resize = Resize(self.scale_range,
                        keep_ratio=True,
                        interpolation=self.interpolation,
                        lazy=False)
        results = resize(results)
        return results


@PIPELINES.register()
class RandomCrop_v2(BaseOperation):
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "imgs" and "img_shape", added or
    modified keys are "imgs", "lazy"; Required keys in "lazy" are "flip",
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    def __init__(self, size: int, lazy: bool = False) -> None:
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    def __call__(self, results: _RESULT) -> _RESULT:
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))
        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(new_crop_quadruple,
                                             dtype=np.float32)

        new_h, new_w = self.size, self.size

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            entity_box_crop = EntityBoxCrop(results['crop_bbox'])
            results = entity_box_crop(results)

        return results


def iminvert(img: np.ndarray) -> np.ndarray:
    """Invert (negate) an image.

    Args:
        img (ndarray): Image to be inverted.

    Returns:
        ndarray: The inverted image.
    """
    return np.full_like(img, 255) - img


@PIPELINES.register()
class Flip(BaseOperation):
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "lazy" and "flip_direction". Required keys in "lazy" is
    None, added or modified key are "flip" and "flip_direction". The Flip
    augmentation should be placed after any cropping / reshaping augmentations,
    to make sure crop_quadruple is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio: float = 0.5,
                 direction: str = 'horizontal',
                 lazy: bool = False) -> None:
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.lazy = lazy

    def __call__(self, results: _RESULT) -> _RESULT:
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        do_flip = np.random.rand() < self.flip_ratio

        results['flip'] = do_flip
        results['flip_direction'] = self.direction

        if not self.lazy:
            if do_flip:
                for i, img in enumerate(results['imgs']):
                    self.im_flip(img, self.direction, inplace=True)
                lt = len(results['imgs'])
            else:
                results['imgs'] = list(results['imgs'])
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = do_flip
            lazyop['flip_direction'] = self.direction

        if 'gt_bboxes' in results and do_flip:
            assert not self.lazy and self.direction == 'horizontal'
            entity_box_flip = EntityBoxFlip(results['img_shape'])
            results = entity_box_flip(results)

        return results


@PIPELINES.register()
class Normalize(BaseOperation):
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (_ARRAY): Mean values of different channels.
        std (_ARRAY): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """
    def __init__(self,
                 mean: _ARRAY,
                 std: _ARRAY,
                 to_bgr: bool = False,
                 adjust_magnitude: bool = False) -> None:
        if not isinstance(mean, Sequence):
            raise TypeError(f'Mean must be _ARRAY, but got {type(mean)}')

        if not isinstance(std, Sequence):
            raise TypeError(f'Std must be _ARRAY, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results: _RESULT) -> _RESULT:
        n = len(results['imgs'])
        h, w, c = results['imgs'][0].shape
        imgs = np.empty((n, h, w, c), dtype=np.float32)
        for i, img in enumerate(results['imgs']):
            imgs[i] = img

        for img in imgs:
            self.im_norm(img,
                         self.mean,
                         self.std,
                         inplace=True,
                         to_bgr=self.to_bgr)

        results['imgs'] = imgs
        results['img_norm_cfg'] = dict(mean=self.mean,
                                       std=self.std,
                                       to_bgr=self.to_bgr)

        return results
