#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2
import itertools

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    pose_to_view_mat3x4,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    remove_correspondences_with_ids, eye3x4)


def _track_all(corner_storage, mats, real_points, intrinsic_mat, tr_params, min_good_points=10):
    recalc = True
    while recalc:
        recalc = False
        for i in range(len(corner_storage)):
            print(f'On frame {i}.')
            frame_corners = corner_storage[i]
            if mats[i] is None:
                idx = frame_corners.ids.squeeze()
                good_idx = np.array(list(map(lambda x: x is not None, real_points[idx])))
                good_frame_points = frame_corners.points[good_idx]
                if len(good_frame_points) >= min_good_points:
                    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                        np.stack(real_points[idx[good_idx]]),
                        good_frame_points, intrinsic_mat,
                        None)
                    if retval:
                        mats[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                        filter_outliers = np.setdiff1d(idx,
                                                       np.intersect1d(inliers.squeeze(), idx))
                        real_points[filter_outliers] = None
                        print(f'Filtered {len(filter_outliers)} points.')
                if mats[i] is not None:
                    recalc = True
                    for j in range(len(corner_storage)):
                        if j != i and mats[j] is not None:
                            corrs = build_correspondences(frame_corners,
                                                          corner_storage[j])
                            points, idx, _ = triangulate_correspondences(corrs, mats[i],
                                                                         mats[j],
                                                                         intrinsic_mat, tr_params)
                            print(
                                f'Triangulated points: {len(points)}. Current PC size: {len(list(filter(lambda x: x is not None, real_points)))}.')
                            for id, point in zip(idx, points):
                                if real_points[id] is None:
                                    real_points[id] = point

    return mats, real_points


def process_transition(intrinsic_mat, tr_params, first_corners, second_corners, corrs_thresh=12):
    corrs = build_correspondences(first_corners, second_corners)
    first_points, second_points = corrs.points_1, corrs.points_2
    if len(corrs.points_1) < corrs_thresh or len(corrs.points_2) < corrs_thresh:
        return 0, None
    em, em_mask = cv2.findEssentialMat(first_points, second_points, intrinsic_mat)
    retval, h_mask = cv2.findHomography(first_points, second_points, method=cv2.RANSAC,
                                        ransacReprojThreshold=tr_params.max_reprojection_error)
    corrs = remove_correspondences_with_ids(corrs, np.where(em_mask.squeeze() == 0)[0])
    if em_mask[em_mask != 0].size < h_mask[h_mask != 0].size:
        return 0, None
    R1, R2, t = cv2.decomposeEssentialMat(em)
    max_result = (-1, -1)
    for rotation, vec in itertools.product([R1, R2], [t, -t]):
        pose = Pose(rotation.T, np.dot(rotation.T, vec))
        points, ids, _ = triangulate_correspondences(corrs, eye3x4(),
                                                     pose_to_view_mat3x4(pose),
                                                     intrinsic_mat,
                                                     tr_params)
        if len(points) > max_result[0]:
            max_result = (len(points), pose)

    return max_result


def get_first_two_views(corner_storage, intrinsic_mat, tr_params):
    best_init = (0, -1, -1)
    first_corners = corner_storage[0]
    for i, corners in enumerate(corner_storage):
        if i == 0:
            continue
        size, pose = process_transition(intrinsic_mat, tr_params, first_corners, corners)
        print(f'Init stage: found {size}')
        if best_init[0] < size or best_init[1] == -1:
            best_init = (size, i, pose)
    return best_init


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    mats = np.repeat(None, len(corner_storage))
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    tr_params = TriangulationParameters(max_reprojection_error=2.3,
                                        min_triangulation_angle_deg=2,
                                        min_depth=0.1)

    if known_view_1 is None or known_view_2 is None:
        best_init = get_first_two_views(corner_storage, intrinsic_mat, tr_params)
        if best_init[1] == -1:
            pass
        else:
            known_view_1 = (0, eye3x4())
            known_view_2 = (best_init[1], best_init[2])
    # TODO: implement
    view_mats, point_cloud_builder = [], PointCloudBuilder()

    mats[known_view_1[0]] = known_view_1[1]
    mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    real_points = np.repeat(None, corner_storage.max_corner_id() + 1)
    corrs = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    points, idx, _ = triangulate_correspondences(corrs, mats[known_view_1[0]], mats[known_view_2[0]],
                                                 intrinsic_mat, tr_params)
    if len(points) == 0:
        pass
    for id, point in zip(idx, points):
        if real_points[id] is None:
            real_points[id] = point

    print(f'Processed first two frames. Found 3d points: {len(points)}.')

    mats, real_points = _track_all(corner_storage, mats, real_points, intrinsic_mat, tr_params)

    # todo check mats
    for i, mat in enumerate(mats):
        if mat is None:
            mats[i] = eye3x4()
            print(f'Some mat is none!')

    print(f'Finished tracking.')

    view_mats = mats
    real_points = list(filter(lambda x: x is not None, real_points))
    point_cloud_ids = np.arange(0, len(real_points))
    point_cloud_builder = PointCloudBuilder(point_cloud_ids, np.array(real_points))

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    print(f'Built point cloud and poses. PC size:{len(point_cloud_ids)}.')
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
