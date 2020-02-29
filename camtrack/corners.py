#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from numpy.linalg import norm

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _process_first_frame(image_1, CV_GFTT_PARAMS, BSIZE):
    first_corners = cv2.goodFeaturesToTrack(image_1, *CV_GFTT_PARAMS)
    corners = FrameCorners(
        np.array(list(range(len(first_corners)))),
        first_corners,
        np.array(np.repeat(BSIZE, len(first_corners)))
    )
    return len(first_corners), corners


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    corners = FrameCorners(
        np.array([0]),
        np.array([[0, 0]]),
        np.array([55])
    )
    builder.set_corners_at_frame(0, corners)

    FILTER_THRESH = 10

    CV_GFTT_MAX_CORNERS = 500
    CV_GFTT_QUALITY_LVL = 0.001
    CV_GFTT_MIN_DIST = 10
    CV_GFTT_CORNERS = None
    CV_GFTT_MASK = None
    CV_GFTT_BSIZE = 10
    CV_GFTT_HARRIS = False
    CV_GFTT_K = 0.05

    CV_OFPLK_NEXT_POINTS = None

    CV_GFTT_PARAMS = [CV_GFTT_MAX_CORNERS, CV_GFTT_QUALITY_LVL,
                      CV_GFTT_MIN_DIST, CV_GFTT_CORNERS, CV_GFTT_MASK,
                      CV_GFTT_BSIZE, CV_GFTT_HARRIS, CV_GFTT_K]
    CV_OFPLK_KPARAMS = dict(winSize=(11, 11))

    current_id, current_corners = _process_first_frame(image_0, CV_GFTT_PARAMS, CV_GFTT_BSIZE)
    builder.set_corners_at_frame(0, current_corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        cv2_corners = cv2.goodFeaturesToTrack(image_1, *CV_GFTT_PARAMS)
        image_0_mono = np.round((image_0 * 255)).astype('uint8')
        image_1_mono = np.round((image_1 * 255)).astype('uint8')
        features, status, track_error = cv2.calcOpticalFlowPyrLK(image_0_mono, image_1_mono,
                                                                 current_corners.points.reshape(-1, 1, 2).astype(
                                                                     'float32'),
                                                                 CV_OFPLK_NEXT_POINTS,
                                                                 **CV_OFPLK_KPARAMS)
        status = status.squeeze()
        mask_good = (status == 1)
        features, current_ids = features[mask_good], current_corners.ids[mask_good]
        cv2_corners = np.array(list(filter(lambda x: np.min(norm(features - x, axis=-1)) > FILTER_THRESH, cv2_corners)))
        cv2_corners_ids = np.arange(current_id, current_id + len(cv2_corners))
        current_id = current_id + len(cv2_corners)
        corners = FrameCorners(
            np.concatenate([current_ids.squeeze(), cv2_corners_ids]),
            np.concatenate([features.reshape(-1, 1, 2), cv2_corners.reshape(-1, 1, 2)]),
            np.array(np.repeat(10, len(features) + len(cv2_corners)))
        )

        builder.set_corners_at_frame(frame, corners)
        current_corners = corners
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
