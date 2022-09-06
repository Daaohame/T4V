import sys

import torchvision.transforms as T
from gluoncv.torch.data.transforms.videotransforms import (
    video_transforms,
    volume_transforms,
)
import decord   # must be imported after gluoncv


def read_video_clip(video_path, interval=2):
    vr = decord.VideoReader(video_path)
    frame_id_list = list(range(0, min(len(vr), 64), interval))
    video_data = vr.get_batch(frame_id_list).asnumpy()
    return video_data


def video_transform_gluoncv(video_data, device=None):
    """Note: video_data should be PIL Image or arrays, [n_frames, H, W, C]"""
    crop_size = 224
    # short_side_size = min(np.shape(video_data)[1], np.shape(video_data)[2])
    # crop_size = min(short_side_size, crop_size)
    short_side_size = 256
    transform_fn = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(crop_size, crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    clip_input = transform_fn(video_data)
    if device is not None:
        clip_input = transform_fn(video_data).to(device)
    return clip_input


def video_transform(video_data, device=None):
    """Note: video_data should be Tensor, [n_frames, C, H, W]"""
    crop_size = 224
    # short_side_size = min(np.shape(video_data)[1], np.shape(video_data)[2])
    # crop_size = min(short_side_size, crop_size)
    short_side_size = 256
    transform_fn = T.Compose(
        [
            T.Resize(
                short_side_size,
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.CenterCrop(size=(crop_size, crop_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    clip_input = transform_fn(video_data).transpose(0, 1)
    if device is not None:
        clip_input = transform_fn(video_data).to(device)
    return clip_input