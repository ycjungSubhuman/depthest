import sys
import getopt
import shutil
import os
import cv2
import re
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from functools import partial

"""
-------------------------------------------------------------------------------
Utility functions that most functions depend on
-------------------------------------------------------------------------------
"""


def prepare_root_inter(root_inter):
    """
    Create and prepare directory for intermediate images

    mkdir and clear all files

    Side Effect)
    May delete all contents of a derectory or create a new directory 
    in your file system

    Args)
    root_inter          root directory for intermedinate images
    """
    if not root_inter:
        return None

    if os.path.exists(root_inter):
        shutil.rmtree(root_inter)
    os.makedirs(root_inter, exist_ok=True)

    return root_inter


"""
-------------------------------------------------------------------------------
Camera intrinsic parameter calibaration
-------------------------------------------------------------------------------
"""


def get_calibration_sample_paths(root_sample):
    """
    Get a list of sample image paths

    All images are stored without any file name rules.
    All images with '.png' or '.jpg' format under 'root_sample'
    will be returned

    Args)
    root_sample         sample root directory path

    Returns)
    A list of paths. list(string)
    """

    return [os.path.join(root_sample, path) for path
            in os.listdir(root_sample)
            if path.endswith('.jpg') or path.endswith('.png')]


def get_calibration_samples(root_sample):
    """
    Get a list of sample images

    All images are stored without any file name rules.
    All images with '.png' or '.jpg' format under 'root_sample'
    will be returned

    Args)
    root_sample         sample root directory path

    Returns)
    A list of BGR images. list(numpy.ndarray(dtype=np.uint8))
    """

    paths = get_calibration_sample_paths(root_sample)

    def check_image(li):
        """
        Image sanity check
            - Uniform image dimension
            - Existance
        """
        size = None
        for i, elem in enumerate(li):
            if elem is None:
                raise Exception('Cannot load some image: {}'.format(paths[i]))

            if size is None:
                size = elem.shape[:2]
            elif elem.shape[:2] != size:
                raise Exception(
                    'Some images have different shape: {}. Prev: {}, Curr: {}'
                    .format(paths[i], size, elem.shape[:2]))

    result = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]
    check_image(result)
    return result


def preprocess_calibration_sample(
        image, path_inter, square_size,
        checker_rows, checker_cols):
    """
    Detect corners from a checkerboard image

    Args)
    image               single-channel checkerboard image
    path_inter          root directory for intermediate results
    square_size         length of one side of a square in the chessboard
                        in millimeter.
    checker_rows        number of checkers in rows
    checker_cols        number of checkers in columns
    """
    found, corners = cv2.findChessboardCorners(
        image, (checker_rows, checker_cols))

    if found:
        # Refine corner positions
        criteria_term = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            1e-4)  # (maxcount,eps)
        cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), criteria_term)
    else:
        return None

    if path_inter:
        # Draw intermediate detection results
        img_inter = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(
            img_inter, (checker_rows, checker_cols), corners, found)
        cv2.imwrite(path_inter, img_inter)

    return corners.reshape(-1, 2)


def calibrate(root_sample, root_inter, square_size,
              checker_rows, checker_cols):
    """
    Get camera intrinsic matrix from calibration samples

    Args)
    root_sample         sample root directory
    root_inter          intermediate result root directory
    square_size         length of one side of a square in the chessboard. 
                        in millimeter.
    checker_rows        number of checkers in rows
    checker_cols        number of checkers in columns

    Returns)
    (camera intrinsic matrix K, rms error).
        (numpy.ndarray(shape=(3,3), dtype=np.float32), float)
    """
    def get_pts_checker():
        """
        Get grid point positions in millimeters unit (x, y, 0)
        """
        pts = np.zeros((checker_rows * checker_cols, 3), dtype=np.float32)
        pts[:, :2] = np.indices((checker_rows, checker_cols)).T.reshape(-1, 2)
        pts *= square_size
        return pts

    corners_ref = get_pts_checker()
    samples = get_calibration_samples(root_sample)

    with ProcessPoolExecutor() as exec:
        path_inters = [os.path.join(
            root_inter, '{:06d}.png'.format(n)) for n in range(len(samples))]
        corners = list(exec.map(preprocess_calibration_sample, samples,
                                path_inters,
                                repeat(square_size, len(samples)),
                                repeat(checker_rows, len(samples)),
                                repeat(checker_cols, len(samples))))
    corners_detected = [c for c in corners if c is not None]

    print('{:d} out of {:d} images are valid'.format(
        len(corners_detected), len(corners)))

    rms, K, _, _, _ = cv2.calibrateCamera(
        [corners_ref for _ in range(len(corners_detected))], corners_detected,
        (samples[0].shape[0], samples[0].shape[1]), None, None)

    return rms, K


def main_camera_intrinsic():
    """
    Calibrate images from checkerboard samples

    usage:
        camera_intrinsic.py [--root_sample <path>] [--root_inter <path>]
            [--square_size <float> (millimeter)]
            [--checker_rows <int>] [--checker_cols <int>]
    """
    ROOT_SAMPLE_DEFAULT = 'data/checkerboard'
    ROOT_INTER_DEFAULT = 'inter/intrinsic'
    SQUARE_SIZE_DEFAULT = 20.0
    CHECKER_ROWS_DEFAULT = 7
    CHECKER_COLS_DEFAULT = 9

    def get_options():
        """
        Parse command line arguments
        """
        args, _ = getopt.gnu_getopt(sys.argv[1:], '', [
                                'root_sample=', 'root_inter=', 'square_size=',
                                'checker_rows=', 'checker_cols='])
        args = dict(args)
        args.setdefault('--root_sample', ROOT_SAMPLE_DEFAULT)
        args.setdefault('--root_inter', ROOT_INTER_DEFAULT)
        args.setdefault('--square_size', SQUARE_SIZE_DEFAULT)
        args.setdefault('--checker_rows', CHECKER_ROWS_DEFAULT)
        args.setdefault('--checker_cols', CHECKER_COLS_DEFAULT)

        return args

    options = get_options()
    rms, K = calibrate(
        root_sample=options['--root_sample'],
        root_inter=prepare_root_inter(options['--root_inter']),
        square_size=float(options['--square_size']),
        checker_rows=int(options['--checker_rows']),
        checker_cols=int(options['--checker_cols']))

    print('Intrinsic matrix K: ')
    print(K)
    print('RMS: {}'.format(rms))
    np.save(os.path.join(options['--root_sample'], 'K.npy'), K)


"""
-------------------------------------------------------------------------------
Feature Matching 
-------------------------------------------------------------------------------
"""


class SIFTBootleg(cv2.Feature2D):
    """
    A bootleg version of SIFT feature detector.
    """

    def __init__(self, *args, **kwargs):
        pass


class ORBWrapper(cv2.Feature2D):
    """
    A wrapper for OpenCV ORB feature detector

    Note)
    The purpose of this wrapper is to match the signatures for the initializer
    with the SIFT class
    """

    def __init__(self, *args, **kwargs):
        self = cv2.ORB_create(*args, **kwargs)


def extract_feature(image1, image2, desc):
    """
    Given two images, extract all features from detected keypoints of
    image1 and image2

    Args)
    image1              an argibtary single-channel image.
    image2              another single-channel image, must contain similar 
                        contents to image1
    desc                A cv2.Feature2D instance for detector instantiation

    Returns)
    (key_image1, key_image2, feat_image1, feat_image2).
        (list(cv2.KeyPoint), list(cv2.KeyPoint), 
         np.ndarray(shape=(len(key),n), dtype=np.uint8), 
         np.ndarray(shape=(len(key),n), dtype=np.uint8))
         where n is the size of a single feature created by 'type_desc'
    """
    key_image1, feat_image1 = desc.detectAndCompute(image1, None)
    key_image2, feat_image2 = desc.detectAndCompute(image2, None)
    return key_image1, key_image2, feat_image1, feat_image2


def is_good_match(thres_lowe, match):
    """
    Returns a if a match is good using lowe test
    """
    match_top, match_second = match
    return match_top.distance < thres_lowe * match_second.distance


def match_correspondence(
        feat1, feat2,
        id_type_matcher=cv2.DescriptorMatcher_FLANNBASED,
        thres_lowe=0.7):
    """
    Compute correspondence between two images

    Args)
    feat1, feat2                np.ndarray(shape=(len(key),n), dtype=np.uint8).
                                where n is the size of a single feature
    id_type_matcher             a value for cv2.DescriptorMatcher_create
    thres_lowe                  a ratio threshold for lowe test

    Returns)
    a list of length-2 list of cv2.DMatch
    """
    matcher = cv2.DescriptorMatcher_create(id_type_matcher)
    matches = matcher.knnMatch(
        feat1.astype(np.float32), feat2.astype(np.float32), 2)
    good_matches = []
    for m1, m2 in matches:
        if is_good_match(thres_lowe, (m1, m2)):
            good_matches.append(m1)
    return good_matches


def visualize_correspondence(image1, image2, key1, key2, matches):
    """
    Visualize correspondences of two images by drawing keypoints on each
    image and drawing lines across corresponding keypoints

    Args)
    image1              np.ndarray(shape=(w1,h1), dtype=np.uint8)
    image2              np.ndarray(shape=(w2,h2), dtype=np.uint8)
    key1                keypoints for image1. list(cv2.KeyPoint)
    key2                keypoints for image2. list(cv2.KeyPoint)
    matches             correspondences, list(list(DMatch, length=2))

    Returns)
    A visualization image.
        np.ndarray(shape=(w1+w2,max(h1,h2)), dtype=np.uint8)
    """
    width_max = max(image1.shape[0], image2.shape[0])
    height_sum = image1.shape[1] + image2.shape[1]
    result = np.zeros((width_max, height_sum, 3), dtype=np.uint8)
    cv2.drawMatches(image1, key1, image2, key2, matches, result)

    return result


def main_feature_matching():
    ROOT_SAMPLE_DEFAULT = 'data/scene'
    ROOT_INTER_DEFAULT = 'inter/match'
    NUM_KEYPOINT_DEFAULT = 500
    THRES_LOWE_DEFAULT = 0.7
    TYPE_DESC_DEFAULT = 'ORB'

    def get_options():
        """
        Parse command line arguments
        """
        args, _ = getopt.getopt(sys.argv[1:], '', [
                                'root_sample=', 'root_inter=',
                                'thres_lowe=', 'num_keypoint=',
                                'type_desc='])
        args = dict(args)
        args.setdefault('--root_sample', ROOT_SAMPLE_DEFAULT)
        args.setdefault('--root_inter', ROOT_INTER_DEFAULT)
        args.setdefault('--thres_lowe', THRES_LOWE_DEFAULT)
        args.setdefault('--num_keypoint', NUM_KEYPOINT_DEFAULT)
        args.setdefault('--type_desc', TYPE_DESC_DEFAULT)

        return args

    def get_all_scene_images(root_sample):
        """
        Read all image pairs in root_sample

        Note)
        image pairs should have filename
            'scene%d_0.jpg', 'scene%d_1.jpg'

        Returns)
        list((np.ndarray(shape=(w,h), dtype=np.uint8), 
              np.ndarray(shape=(w,h), dtype=np.uint8))
        """
        i = 0
        pair_images = []
        while True:
            path_left = os.path.join(
                root_sample, 'scene{}_{}.jpg'.format(i, 0))
            path_right = os.path.join(
                root_sample, 'scene{}_{}.jpg'.format(i, 1))

            exists_pair = os.path.exists(
                path_left) and os.path.exists(path_right)
            if not exists_pair:
                break
            pair_images.append((cv2.imread(path_left, cv2.IMREAD_GRAYSCALE),
                                cv2.imread(path_right, cv2.IMREAD_GRAYSCALE)))
            i += 1
        return pair_images

    options = get_options()
    prepare_root_inter(options['--root_inter'])
    samples = get_all_scene_images(options['--root_sample'])

    if options['--type_desc'] == 'ORB':
        desc = cv2.ORB_create(nfeatures=int(options['--num_keypoint']))
    elif options['--type_desc'] == 'SIFT':
        desc = SIFTBootleg(nfeatures=int(options['--num_keypoint']))
    else:
        raise Exception('Unidentified type_desc: {}'.format(
            options['--type_desc']))

    for i, (image_left, image_right) in enumerate(samples):
        key1, key2, feat1, feat2 = extract_feature(
            image_left, image_right, desc)
        matches = match_correspondence(
            feat1, feat2, thres_lowe=float(options['--thres_lowe']))
        if options['--root_inter']:
            image_vis = visualize_correspondence(
                image_left, image_right, key1, key2, matches)
            cv2.imwrite(os.path.join(
                options['--root_inter'], '{:04d}.jpg'.format(i)), image_vis)


"""
-------------------------------------------------------------------------------
Camera extrinsic parameter optimization
-------------------------------------------------------------------------------
"""


"""
-------------------------------------------------------------------------------
main
-------------------------------------------------------------------------------
"""

if __name__ == '__main__':
    # main_camera_intrinsic()
    main_feature_matching()
