from typing import Any, Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image

# ----------------------------------------------------------------
# Common variable alias names
# ----------------------------------------------------------------
_IMSIZE = Tuple[int, int]
_SCALE = Union[int, float, Tuple[int, int]]
_IMTYPE = Union[np.ndarray, Image.Image, paddle.Tensor]
_BOX = Tuple[int, int, int, int]
_ARRAY = Sequence[Union[int, float]]
_RESULT = Dict[str, Any]

# ----------------------------------------------------------------
# Common interpolation constants
# ----------------------------------------------------------------
PILLOW_INTERP_CODES = {
    "none": Image.NONE,
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "linear": Image.LINEAR,
    "bicubic": Image.BICUBIC,
    "cubic": Image.CUBIC,
    "box": Image.BOX,
    "lanczos": Image.LANCZOS,
    "hamming": Image.HAMMING,
    "antialias": Image.ANTIALIAS
}
OPENCV_INTERP_CODES = {
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

# ----------------------------------------------------------------
# Common flip constants
# ----------------------------------------------------------------
PILLOW_FLIP_CODES = {
    "horizontal": Image.FLIP_LEFT_RIGHT,
    "vertical": Image.FLIP_TOP_BOTTOM
}
CV2_FLIP_CODES = {"horizontal": 1, "vertical": 0}
TENSOR_FLIP_CODES = {"horizontal": 0, "vertical": 1}

# ----------------------------------------------------------------
# Temporarily useless note
# ----------------------------------------------------------------
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


# ----------------------------------------------------------------
# Basic class for common operations on image(s)
# ----------------------------------------------------------------
class BaseOperation(object):
    def __init__(self) -> None:
        super(BaseOperation, self).__init__()

    @staticmethod
    def isPILImage(x: _IMTYPE) -> bool:
        return isinstance(x, Image.Image)

    @staticmethod
    def isNumpy(x: _IMTYPE) -> bool:
        return isinstance(x, np.ndarray)

    @staticmethod
    def isTensor(x: _IMTYPE) -> bool:
        return isinstance(x, paddle.Tensor)

    @staticmethod
    def _compute_length(size: _IMSIZE, scale_factor: Union[int,
                                                           float]) -> _IMSIZE:
        """Compute the new scale size by input size and scale factor.

        Args:
            size (_IMSIZE): input size, including width and height.
            scale_factor (Union[int, float]): scale factor.
        """
        if len(size) != 2:
            raise ValueError(f"len(size) must be 2, but got {len(size)}")
        w, h = size
        new_w = int(w * float(scale_factor) + 0.5)
        new_h = int(h * float(scale_factor) + 0.5)
        return (new_w, new_h)

    def get_scaled_size(self,
                        old_size: _IMSIZE,
                        scale: _SCALE,
                        return_scale: bool = False
                        ) -> Union[_IMSIZE, Tuple[_IMSIZE, _SCALE]]:
        """Get scaled size by old_size and scale param(factor or int size),
           called only when need keep ratio.

        Args:
            old_size (IMSIZE): original size, contains (w, h).
            scale (_SCALE): scale params.
            return_scale (bool, optional): Whether return scale \
                factor together with scale size. Defaults to False.

        Returns:
            Union[_IMSIZE, Tuple[_IMSIZE, _SCALE]]: new size calculated from old size and scale param,
            scale factor followed if specified.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f"scale must be positive, but got {scale}.")
            scale_factor = scale
        elif isinstance(scale, tuple):
            long_side = max(scale)
            short_side = min(scale)
            scale_factor = min(
                long_side / max(h, w),  # Infinity invalidates the long side
                short_side / min(h, w))
        else:
            raise TypeError(
                f"type of scale must be {_SCALE}, but got {type(scale)}.")

        new_size = self._compute_length((w, h), scale_factor)

        if return_scale:
            return (new_size, scale_factor)
        else:
            return new_size

    def im_resize(self,
                  img: _IMTYPE,
                  size: _IMSIZE,
                  interpolation: str,
                  to_numpy: bool = False) -> _IMTYPE:
        """Apply resize function to input image(s).

        Args:
            img (_IMAGE): input image(s).
            size (_IMSIZE): target size which resized to, which contains (w, h).
            interpolation (str): interpolation method for resize function.
            to_numpy (bool, optional): Whether convert to numpy.ndarray after resize. Defaults to False.

        Returns:
            _IMAGE: Resized images(s).
        """
        if self.isNumpy(img):
            return cv2.resize(src=img,
                              dsize=size,
                              interpolation=OPENCV_INTERP_CODES[interpolation])
        elif self.isPILImage(img):
            img = img.resize(size=size,
                             resample=PILLOW_INTERP_CODES[interpolation])
            if to_numpy:
                return np.array(img)
            return img
        elif self.isTensor(img):
            if img.ndim != 4:
                raise ValueError(
                    f"tensor must be 4 dim when resize, but got {img.ndim}.")
            # TODO: Only support 'NCHW' format currently!
            return F.interpolate(
                img,  # [*,*,h,w]
                size=size[::-1],  # (w,h) to (h,w)
                mode=TENSOR_INTERP_CODES[interpolation],
                data_format="NCHW",
                align_corners=False)
        else:
            raise TypeError(
                f"input images must be {_IMTYPE}, but got {type(img)}.")

    def im_flip(self,
                img: _IMTYPE,
                direction: str = "horizontal",
                inplace: bool = False) -> _IMTYPE:
        """Apply flip function to input image(s).

        Args:
            img (_IMAGE): input image(s).
            direction (str, optional): Direction of flip op. Defaults to "horizontal".
            inplace (bool, optional): Whether use inplace op when flip(if available). Defaults to False.

        Returns:
            _IMAGE: Fliped image(s).
        """
        if self.isNumpy(img):
            if inplace:
                return cv2.flip(src=img,
                                flipCode=CV2_FLIP_CODES[direction],
                                dst=img)
            else:
                return cv2.flip(src=img, flipCode=CV2_FLIP_CODES[direction])
        elif self.isPILImage(img):
            return img.transpose(PILLOW_FLIP_CODES[direction])
        elif self.isTensor(img):
            if img.ndim != 4:
                raise ValueError(
                    f"tensor must be 4 dim when resize, but got {img.ndim} dim."
                )
            # TODO: Only support '**HW' format currently!
            return paddle.flip(img, axis=TENSOR_FLIP_CODES[direction])
        else:
            raise TypeError(
                f"input images must be {_IMTYPE}, but got {type(img)}.")

    def im_crop(self, img: _IMTYPE, box: _BOX) -> _IMTYPE:
        """Apply crop function to input image(s).

        Args:
            img (_IMAGE): input image(s).
            box (_BOX): coords of crop box, which is (left, top, right, bottom).

        Returns:
            _IMAGE: Croped img.
        """
        left, top, right, bottom = box
        if self.isNumpy(img):
            return img[top:bottom, left:right]
        elif self.isPILImage(img):
            return img.crop((left, top, right, bottom))
        elif self.isTensor(img):
            # TODO: Only support '**HW' format currently!
            return img[:, :, top:bottom, left:right]
        else:
            raise TypeError(
                f"input images must be numpy.ndarray or PIL.Image.Image or "
                f"paddle.Tensor, but got {type(img)}.")

    def im_norm(self,
                img: np.ndarray,
                mean: np.ndarray,
                std: np.ndarray,
                inplace: bool = False,
                to_bgr: bool = False) -> np.ndarray:
        """Apply normalization to an single image.

        Args:
            img (np.ndarray): input image, image.ndim must be 3.
            mean (np.ndarray): mean value array to subtract.
            std (np.ndarray): std value to divide.
            inplace (bool, optional): Whether use inplace op when normlize(if available). Defaults to False.
            to_bgr (bool, optional): Whether to convert channels from RGB to BGR(inplace, only supprt with cv2). Default to False.

        Returns:
            np.ndarray: Normalized image(s).
        """
        if self.isNumpy(img):
            h, w, c = img.shape
            if to_bgr:
                if img.dtype != np.uint8:
                    raise TypeError(
                        f"img's data type must be uint8, but got {img.dtype}.")
                if c != 3:
                    raise ValueError(
                        f"expect the channels to be 3 when to_bgr=True, but channels of img is [{c}]."
                    )
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace convert
            if inplace:
                if img.dtype != np.uint8:
                    raise TypeError(
                        f"img's data type must be uint8, but got {img.dtype}.")
                if c != 3:
                    raise ValueError(
                        f"expect the channels to be 3 when inplace=True, but channels of img is [{c}]."
                    )
                mean = mean.reshape(1, -1).astype("float64")  # [1, 3]
                std_inv = 1 / (std.reshape(1, -1).astype("float64"))  # [1, 3]
                cv2.subtract(img, mean, img)
                cv2.multiply(img, std_inv, img)
                return img
            else:
                norm_img = img
                norm_img -= mean
                norm_img /= std
                return norm_img
        else:
            raise TypeError(
                f"input images must be numpy.ndarray, but got {type(img)}.")

    def im_stack(self,
                 imgs: Sequence[_IMTYPE],
                 axis: int = 0) -> Union[paddle.Tensor, np.ndarray]:
        """Apply image stack on tuple or list of images.

        Args:
            imgs (Sequence[_IMTYPE]): Sequence of images.
            axis (int, optional): stack axis. Defaults to 0.

        Returns:
            Union[paddle.Tensor, np.ndarray]: Stacked images.
        """
        if isinstance(imgs, (list, tuple)):
            if self.isTensor(imgs[0]):
                return paddle.stack(imgs, axis=axis)
            elif self.isPILImage(imgs[0]) or self.isNumpy(imgs[0]):
                return np.stack(imgs, axis=axis)
            else:
                raise ValueError(
                    f"type of element in imgs must be {_IMTYPE}, but got {type(imgs[0])}."
                )
        else:
            raise ValueError(
                f"type of imgs must be {Sequence[_IMTYPE]}, but got {type(imgs)}."
            )

    def get_im_size(self, img: Union[_IMTYPE, List[_IMTYPE]]) -> _IMSIZE:
        """Get Image size from a single image or Sequence of images.

        Args:
            img (Union[_IMTYPE, List[_IMTYPE]]): a single image or Sequence of images.

        Returns:
            _IMSIZE: (width, height) of a single image.
        """
        if self.isTensor(img):
            h, w = img.shape[-2:]
        elif isinstance(img, list):
            if self.isNumpy(img[0]):
                h, w = img[0].shape[:2]
            elif self.isPILImage(img[0]):
                w, h = img[0].size
            else:
                raise TypeError(
                    f"img must be type of {paddle.Tensor} or {List[Union[np.ndarray, Image.Image]]}, but got {type(img)}."
                )
        else:
            raise TypeError(
                f"img must be type of {paddle.Tensor} or {List[Union[np.ndarray, Image.Image]]}, but got {type(img)}."
            )
        return (w, h)

    def __repr__(self) -> str:
        """Return the representation string of an operation.

        Returns:
            str: representation string.
        """
        repr_str = self.__class__.__name__
        repr_str += "("
        attrs = vars(self)
        for attr_name, attr_value in attrs.items():
            repr_str += f"\n    {attr_name}={attr_value}"
        repr_str += "\n)"
        return repr_str
