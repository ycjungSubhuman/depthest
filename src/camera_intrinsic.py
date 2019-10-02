"""
Camera intrinsic parameter calibaration module

Yuchoel Jung 2019 all rights reserved
"""
import os
import cv2

import numpy as np

ROOT_SAMPLE_DEFAULT = 'data/checkerboard'
ROOT_INTER_DEFAULT = 'inter'
SQUARE_SIZE_DEFAULT = 20.0
CHECKER_ROWS_DEFAULT = 9
CHECKER_COLS_DEFAULT = 6


def get_calibration_sample_paths(root_sample):
    """
    Get a list of sample image paths

    All images are stored without any file name rules.
    All images with '.png' or '.jpg' format under 'root_sample'
    will be returned

    Args)
    root_sample         sample root

    Returns)
    A list of paths. list(string)
    """

    return [os.path.join(root_sample, path) for path
            in os.listdir(root_sample)]


def get_calibration_samples(root_sample):
    """
    Get a list of sample images

    All images are stored without any file name rules.
    All images with '.png' format under 'root_sample'
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
        image, path_inter='', square_size=SQUARE_SIZE_DEFAULT,
        checker_rows=CHECKER_ROWS_DEFAULT, checker_cols=CHECKER_COLS_DEFAULT):
    """
    Detect corners from a checkerboard image

    Args)
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


def calibrate(
        root_sample=ROOT_SAMPLE_DEFAULT, root_inter=ROOT_INTER_DEFAULT,
        square_size=SQUARE_SIZE_DEFAULT, checker_rows=CHECKER_ROWS_DEFAULT,
        checker_cols=CHECKER_COLS_DEFAULT):
    """
    Get camera intrinsic matrix from calibration samples

    Args)
    root_sample         sample root directory path

    Returns)
    (camera intrinsic matrix K, rms error). (numpy.ndarray(shape=(3,3), dtype=np.float32), float)
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
    corners = [preprocess_calibration_sample(
        sample) for sample in get_calibration_samples(root_sample)]
    corners_detected = [c for c in corners if c]

    print('{:d} of of {:d} images are valid'.format(
        len(corners_detected), len(corners)))

    rms, K, _, _, _ = cv2.calibrateCamera(
        [corners_ref for _ in range(len(corners_detected))], corners_detected, None, None)

    return rms, K


def main():
    """
    Calibrate images from checkerboard samples

    usage:
        camera_intrinsic.py [--root_sample <path>] [--root_inter <path>]
            [--square_size <float> (millimeter)]
            [--checker_rows <int>] [--checker_cols <int>]
    """
    import sys
    import getopt
    import shutil

    def get_options():
        """
        Parse command line arguments
        """
        args, _ = getopt.getopt(sys.argv[1:], '', [
                                'root_sample=', 'root_inter=', 'square_size=',
                                'checker_rows=', 'checker_cols='])
        args = dict(args)
        args.setdefault('--root_sample', ROOT_SAMPLE_DEFAULT)
        args.setdefault('--root_inter', ROOT_INTER_DEFAULT)
        args.setdefault('--square_size', SQUARE_SIZE_DEFAULT)
        args.setdefault('--checker_rows', CHECKER_ROWS_DEFAULT)
        args.setdefault('--checker_cols', CHECKER_COLS_DEFAULT)

        return args

    def prepare_root_inter(root_inter):
        """
        Create and prepare directory for intermediate images

        mkdir and clear all files

        Side Effect)
        May delete all contents of a derectory or create a new directory in your file system

        Args)
        root_inter          root directory for intermedinate images
        """
        if not root_inter:
            return None

        if os.path.exists(root_inter):
            shutil.rmtree(root_inter)
        os.makedirs(root_inter, exist_ok=True)

        return root_inter

    options = get_options()
    K, rms = calibrate(
        root_sample=options['--root_sample'],
        root_inter=prepare_root_inter(options['--root_inter']),
        square_size=float(options['--square_size']),
        checker_rows=int(options['--checker_rows']),
        checker_cols=int(options['--checker_cols']))

    print('Intrinsic matrix K: ')
    print(K)
    print('RMS: {}'.format(rms))


if __name__ == '__main__':
    main()
