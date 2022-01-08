from typing import Any, Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image

_IMSIZE = Tuple[int, int]
_SCALE = Union[int, float, Tuple[int, int]]
_IMTYPE = Union[np.ndarray, Image.Image, paddle.Tensor]
_BOX = Tuple[int, int, int, int]
_ARRAY = Sequence[Union[int, float]]
_RESULT = Dict[str, Any]

PILLOW_INTERP_CODES = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "box": Image.BOX,
    "lanczos": Image.LANCZOS,
    "hamming": Image.HAMMING,
    "antialias": Image.ANTIALIAS
}
CV2_INTERP_CODES = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4
}
TENSOR_INTERP_CODES = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "linear": "linear",
    "area": "area",
}

PILLOW_FLIP_CODE = {
    "horizontal": Image.FLIP_LEFT_RIGHT,
    "vertical": Image.FLIP_TOP_BOTTOM
}
CV2_FLIP_CODES = {"horizontal": 1, "vertical": 0}
TENSOR_FLIP_CODES = {"horizontal": 0, "vertical": 1}

# def batch_enable(operation_func):
#     @functools.wraps(operation_func)
#     def batch_apply(*args, **kwargs):
#         print(args[0])
#         return operation_func(*args, **kwargs)
#     return batch_apply

# def varname(p):
#     for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
#         m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
#         if m:
#             return m.group(1)


class BaseOperation(object):
    def __init__(self) -> None:
        super().__init__()

    def _calc_length(self, size: _IMSIZE,
                     scale_factor: Union[int, float]) -> _IMSIZE:
        """Get new scaled size by input size and scale_factor factor

        Args:
            size (_IMSIZE): input size, including width and height
            scale_factor (Union[int, float]): scale factor
        """
        assert len(size) == 2, \
            f"len(size) must be 2, but got {len(size)}"
        w, h = size
        new_w = int(w * float(scale_factor) + 0.5)
        new_h = int(h * float(scale_factor) * 0.5)
        return (new_w, new_h)

    def get_scaled_size(self,
                        old_size: _IMSIZE,
                        scale: _SCALE,
                        return_scale: bool = False
                        ) -> Union[_IMSIZE, Tuple[_IMSIZE, _SCALE]]:
        """Get scaled size by old_size and scale param(factor or int size),
           only used when keep ratio.

        Args:
            old_size (IMSIZE): original size, contains (w, h)
            scale (_SCALE): scale params
            return_scale (bool, optional): Whether return scale \
                factor together with scale size. Defaults to False.

        Returns:
            Union[_IMSIZE, Tuple[_IMSIZE, _SCALE]]: new size calculated from old size and scale param,
            scale factor followed if specified.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f"Invalid scale {scale}, must be positive.")
            scale_factor = scale
        elif isinstance(scale, tuple):
            long_side = max(scale)
            short_side = min(scale)
            scale_factor = min(
                long_side / max(h, w),  # Infinity invalidates the long side
                short_side / min(h, w))
        else:
            raise TypeError(
                f"Scale must be a number or tuple of int, but got {type(scale)}"
            )

        new_size = self._calc_length((w, h), scale_factor)

        if return_scale:
            return new_size, scale_factor
        else:
            return new_size

    def im_resize(self,
                  img: _IMTYPE,
                  size: _IMSIZE,
                  interpolation: str,
                  to_numpy: bool = False) -> _IMTYPE:
        """Apply resize function to input image(s)

        Args:
            img (_IMAGE): input image(s)
            size (_IMSIZE): target size which resized to, which contains (w, h)
            interpolation (str): interpolation method for resize function
            to_numpy (bool, optional): Whether convert to numpy.ndarray after resize. Defaults to False.

        Returns:
            _IMAGE: Resized images(s)
        """
        if isinstance(img, np.ndarray):
            return cv2.resize(src=img,
                              dsize=size,
                              interpolation=CV2_INTERP_CODES[interpolation])
        elif isinstance(img, Image.Image):
            img = img.resize(size=size,
                             resample=PILLOW_INTERP_CODES[interpolation])
            if to_numpy:
                return np.array(img)
            return img
        elif isinstance(img, paddle.Tensor):
            if img.ndim != 4:
                raise ValueError(
                    f"Tensor must be 4 dim when resize, but got {img.ndim}")
            # TODO: Only support 'NCHW' format currently!
            return F.interpolate(
                img,  # [*,*,h,w]
                size=size[::-1],  # (w,h) to (h,w)
                mode=TENSOR_INTERP_CODES[interpolation],
                data_format="NCHW",
                align_corners=False)
        else:
            raise TypeError(
                f"Input images must be numpy.ndarray or PIL.Image.Image or \
                    paddle.Tensor, but got{type(img)}")

    def im_flip(self,
                img: _IMTYPE,
                direction: str = "horizontal",
                inplace: bool = False) -> _IMTYPE:
        """Apply flip function to input image(s)

        Args:
            img (_IMAGE): input image(s)
            direction (str, optional): Direction of flip op. Defaults to "horizontal".
            inplace (bool, optional): Whether use inplace op when flip(if available). Defaults to False.

        Returns:
            _IMAGE: Fliped image(s)
        """
        if isinstance(img, np.ndarray):
            if inplace:
                return cv2.flip(src=img,
                                flipCode=CV2_FLIP_CODES[direction],
                                dst=img)
            else:
                return cv2.flip(src=img, flipCode=CV2_FLIP_CODES[direction])
        elif isinstance(img, Image.Image):
            return img.transpose(PILLOW_FLIP_CODE[direction])
        elif isinstance(img, paddle.Tensor):
            if img.ndim != 4:
                raise ValueError(
                    f"Tensor must be 4 dim when resize, but got {img.ndim}")
            # TODO: Only support '**HW' format currently!
            return paddle.flip(img, axis=TENSOR_FLIP_CODES[direction])
        else:
            raise TypeError(
                f"Input images must be numpy.ndarray or PIL.Image.Image or \
                    paddle.Tensor, but got{type(img)}")

    def im_crop(self, img: _IMTYPE, box: _BOX) -> _IMTYPE:
        """Apply crop function to input image(s)

        Args:
            img (_IMAGE): input image(s)
            box (_BOX): coords of crop box, which is (left, top, right, bottom)

        Returns:
            _IMAGE: Croped img
        """
        left, top, right, bottom = box
        if isinstance(img, np.ndarray):
            return img[top:bottom, left:right]
        elif isinstance(img, Image.Image):
            return img.crop((left, top, right, bottom))
        elif isinstance(img, paddle.Tensor):
            if img.ndim != 4:
                raise ValueError(
                    f"Tensor must be 4 dim when resize, but got {img.ndim}")
            # TODO: Only support '**HW' format currently!
            return img[:, :, top:bottom, left:right]
        else:
            raise TypeError(
                f"Input images must be numpy.ndarray or PIL.Image.Image or \
                    paddle.Tensor, but got{type(img)}")

    def im_norm(self,
                img: _IMTYPE,
                mean: np.ndarray,
                std: np.ndarray,
                inplace: bool = False,
                to_bgr: bool = False) -> _IMTYPE:
        """Apply normalization to input image(s)

        Args:
            img (_IMAGE): input image(s)
            mean (np.ndarray): mean value array to subtract
            std (np.ndarray): std value to divide
            inplace (bool, optional): Whether use inplace op when normlize(if available). Defaults to False.
            to_bgr (bool, optional): Whether to convert channels from RGB to BGR(inplace, and only supprt with cv2). Default to False.
        Returns:
            _IMAGE: Normalized image(s)
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                raise TypeError(
                    f"img.dtype can't be uint8, but got {img.dtype}")
            if to_bgr:
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            if inplace:
                mean = mean.reshape(1, -1).astype("float64")  # [1, 3]
                std_inv = 1 / (std.reshape(1, -1).astype("float64")
                               )  # [1, 3], reciprocal of std
                cv2.subtract(img, mean, img)  # inplace
                cv2.multiply(img, std_inv, img)  # inplace
                return img
            else:
                norm_img = img
                norm_img -= mean
                norm_img /= std
                return norm_img
        elif isinstance(img, paddle.Tensor):
            if img.ndim != 4:
                raise ValueError(
                    f"Tensor must be 4 dim when resize, but got {img.ndim}")
            # TODO: Only support '**HW' format currently!
            norm_imgs = img
            norm_imgs -= mean
            norm_imgs /= std
            return norm_imgs
        else:
            raise TypeError(
                f"Input images must be numpy.ndarray or paddle.Tensor, but got{type(img)}"
            )

    def get_size(self, img: Union[_IMTYPE, List[_IMTYPE]]) -> _IMSIZE:
        if isinstance(img, paddle.Tensor):
            h, w = img.shape[-2:]
        elif isinstance(img, list):
            if isinstance(img[0], np.ndarray):
                h, w = img[0].shape[:2]
            elif isinstance(img[0], Image.Image):
                w, h = img[0].size
            else:
                raise TypeError(
                    f"img must be type of {Union[_IMTYPE, List[_IMTYPE]]}, but got {type(img)}"
                )
        else:
            raise TypeError(
                f"img must be type of {Union[_IMTYPE, List[_IMTYPE]]}, but got {type(img)}"
            )
        return w, h

    def __repr__(self) -> str:
        ret = self.__class__.__name__
        ret += "("
        attrs = vars(self)
        for attr_name, attr_value in attrs.items():
            ret += f"\n  {attr_name}={attr_value}"
        ret += "\n)"
        return ret


op = BaseOperation()

im = np.arange(2 * 2 * 3).reshape([2, 2, 3]).astype("uint8")
print(im)
op.im_flip(im, inplace=True)
print(im)
