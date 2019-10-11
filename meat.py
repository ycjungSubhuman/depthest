import sys
import getopt
import shutil
import os
import cv2
import re
import random
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat, product
from functools import partial

"""
-------------------------------------------------------------------------------
Options for this script
-------------------------------------------------------------------------------
"""
WIDTH_IMAGE = 640
HEIGHT_IMAGE = 480
options = {
    'root_checker_sample': 'data/checkerboard',
    'root_checker_inter': 'inter/intrinsic',
    'square_size': 20.0,
    'checker_rows': 7,
    'checker_cols': 9,

    'root_stereo_sample': 'data/scene',
    'root_match_inter': 'inter/match',
    'thres_lowe': 0.7,
    'num_keypoint': 5000,
    'type_desc': 'ORB',

    'ransac_radius': 0.1,
    'ransac_iter': 200,

    'root_rectify_inter': 'inter/rectify',
}

"""
-------------------------------------------------------------------------------
Utility functions that many functions depend on
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


def normalize_image_size(im):
    """
    Fix image size
    """
    return cv2.resize(im, (WIDTH_IMAGE, HEIGHT_IMAGE), interpolation=cv2.INTER_AREA)


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
        pair_images.append(
            (normalize_image_size(cv2.imread(path_left, cv2.IMREAD_GRAYSCALE)),
                normalize_image_size(cv2.imread(path_right, cv2.IMREAD_GRAYSCALE))))
        i += 1
    return pair_images


def get_all_npy(root_sample, format):
    """
    Read all .npy file in root_sample

    Returns)
    list(np.ndarray)
    """
    i = 0
    keypoint_matches = []
    while True:
        path = os.path.join(root_sample, format.format(i))
        if not os.path.exists(path):
            break
        elem = np.load(path)
        keypoint_matches.append(elem)
        i += 1
    return keypoint_matches


def get_all_keypoint_matches(root_sample):
    """
    Read all keypoint matches in root_sample

    Note)
    keypoint match files should have filename
        'scene%d.npy'

    Returns)
    list(np.ndarray(shape=(N, 2, 2), dtype=np.float32))
    """
    return get_all_npy(root_sample, 'scene{}.npy')


def get_all_F(root_sample):
    """
    Read all fundamental matrices in root_sample

    Note)
    keypoint match files should have filename
        'scene%d_F.npy'

    Returns)
    list(np.ndarray(shape=(3, 3), dtype=np.float32))
    """
    return get_all_npy(root_sample, 'scene{}_F.npy')


def get_all_E(root_sample):
    """
    Read all essential matrices in root_sample

    Note)
    keypoint match files should have filename
        'scene%d_E.npy'

    Returns)
    list(np.ndarray(shape=(3, 3), dtype=np.float32))
    """
    return get_all_npy(root_sample, 'scene{}_E.npy')


def get_all_pose(root_sample):
    """
    Read all pose files in root_sample

    Note)
    keypoint match files should have filename
        'scene%d_pose.npz'

    Returns)
    list(npz)
    """
    return get_all_npy(root_sample, 'scene{}_pose.npz')


def normalize_coordinate(pt):
    """
    Convert image space coordinate to [-1, 1]
    """
    pt = pt.copy()
    pt[0] = pt[0]/WIDTH_IMAGE*2 - 1.0
    pt[1] = pt[1]/HEIGHT_IMAGE*2 - 1.0

    return pt


def preprocess_match(key1, key2, matches):
    """
    Convert two list(cv2.KeyPoint) into normalized coordinate
    list(list(np.ndarray(shape=(2), dtype=np.float32)))
    """
    result = []
    for m in matches:
        pt1 = normalize_coordinate(
            np.array(key1[m.queryIdx].pt, dtype=np.float32))
        pt2 = normalize_coordinate(
            np.array(key2[m.trainIdx].pt, dtype=np.float32))
        result.append([pt1, pt2])
    return result


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

    result = [
        normalize_image_size(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) for path in paths]
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
    (rms error, camera intrinsic matrix K, distortion coefficient).
        (float, 
         numpy.ndarray(shape=(3,3), dtype=np.float32), 
         np.ndarray(shape=(5), dtype=np.float32))
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

    rms, K, distcoeffs, _, _ = cv2.calibrateCamera(
        [corners_ref for _ in range(len(corners_detected))], corners_detected,
        (samples[0].shape[0], samples[0].shape[1]), None, None)

    return rms, K, distcoeffs


def main_camera_intrinsic():
    """
    Calibrate images from checkerboard samples. 
    Print camera intrinsic matrix K and RMS error.
    Saves K matrix under root_checker_sample
    """
    rms, K, distcoeffs = calibrate(
        root_sample=options['root_checker_sample'],
        root_inter=prepare_root_inter(options['root_checker_inter']),
        square_size=float(options['square_size']),
        checker_rows=int(options['checker_rows']),
        checker_cols=int(options['checker_cols']))

    print('Intrinsic matrix K: ')
    print(K)
    print('Distortion coefficients: ')
    print(distcoeffs)
    print('RMS: {}'.format(rms))
    np.save(os.path.join(options['root_checker_sample'], 'K.npy'), K)
    np.save(os.path.join(
        options['root_checker_sample'], 'dist.npy'), distcoeffs)


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
    list(cv2.DMatch)
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
    """
    Perform feature matching and save intermediate results

    Task 1-3
    """
    print('--------------------------------------------------------------------')
    print('Task 1-3')
    print('--------------------------------------------------------------------')
    prepare_root_inter(options['root_match_inter'])
    samples = get_all_scene_images(options['root_stereo_sample'])

    if options['type_desc'] == 'ORB':
        desc = cv2.ORB_create(nfeatures=int(options['num_keypoint']))
    elif options['type_desc'] == 'SIFT':
        desc = SIFTBootleg(nfeatures=int(options['num_keypoint']))
    else:
        raise Exception('Unidentified type_desc: {}'.format(
            options['type_desc']))

    for i, (image_left, image_right) in enumerate(samples):
        key1, key2, feat1, feat2 = extract_feature(
            image_left, image_right, desc)
        matches = match_correspondence(
            feat1, feat2, thres_lowe=float(options['thres_lowe']))
        path = os.path.join(
            options['root_stereo_sample'], 'scene{}.npy'.format(i))
        np.save(path, preprocess_match(key1, key2, matches))
        if options['root_match_inter']:
            image_vis = visualize_correspondence(
                image_left, image_right, key1, key2, matches)
            cv2.imwrite(os.path.join(
                options['root_match_inter'], '{:04d}.jpg'.format(i)), image_vis)


"""
-------------------------------------------------------------------------------
Camera extrinsic parameter optimization
-------------------------------------------------------------------------------
"""


class RegressionProblem:
    """
    Regress a value from samples
    """

    def cost(self, parameter, observation):
        """
        Calculate cost when given a set of parameters and a data point

        Args)
        parameter               model argument
        observation             a single observed data such as a point

        Returns)
        float
        """
        raise NotImplementedError()

    def estimate(self, observation):
        """
        Estimate parameters from observation
        """
        raise NotImplementedError()

    def dof(self):
        """
        Get the degree of freedom
        """
        raise NotImplementedError()

    def dim_parameter(self):
        """
        Get the number of paramters
        """
        raise NotImplementedError()


def get_fundamental_matrix(parameter, rank2=True):
    """
    Create a 3x3 fundamental matrix from a 9-dimension parameter

    Args)
    parameter               np.ndarray(shape=(9), dtype=np.float32)
    rank2                   if true, impose rank-2 constraint
    """
    if rank2:
        F_free = np.matrix(parameter).reshape((3, 3)).T
        u, s_free, vh = np.linalg.svd(F_free)
        s = np.array([s_free[0], s_free[1], 0.0])
        F = u @ np.diag(s) @ vh
    else:
        F = np.matrix(parameter).reshape((3, 3)).T
    return F / F[2, 2]


def homogeneous(x):
    """
    Get homogenous vector
    """
    return np.concatenate((x, np.ones(1, dtype=np.float32)))


def hnormalize(x):
    """
    Convert homogeneous vector to original 
    """
    return x[:-1] / x[-1]


def formulate_8point_matrix(observations):
    """
    Calculate A matrix used in 8-point fundamental 
    matrix calculation

    Args)
    observations             list(np.ndarray(shape=(2,2), dtype=np.float32))
    """
    assert len(observations) == 8
    elems = []
    for observation in observations:
        x1 = observation[0]
        x2 = observation[1]
        elems.append([x2[0]*x1[0], x2[0]*x1[1], x2[0],
                      x2[1]*x1[0], x2[1]*x1[1], x2[1], x1[0], x1[1], 1.0])
    A = np.matrix(elems)

    return A


class CameraExtrinsicProblem(RegressionProblem):
    """
    Given parameters for fundamental metrix F,
    calculate x1.T mm F mm x2
    """

    def cost(self, parameter, observation):
        """
        Args)
        parameter               np.ndarray(shape=(8), dtype=np.float32)
        observation             list(np.ndarray(shape=(2), dtype=np.float32))
                                x1 := observation[0]
                                x1 := observation[1]
        """
        F = get_fundamental_matrix(parameter)
        x1 = np.matrix(homogeneous(observation[0])).T
        x2 = np.matrix(homogeneous(observation[1])).T

        xF = (x2.T @ F).A1
        xFx = (x2.T @ F @ x1)[0, 0]
        Fx = (F @ x1).A1

        cost = xFx*(1.0/(xF[0]**2 + xF[1]**2) + 1.0/(Fx[0]**2 + Fx[1]**2))
        return cost

    def estimate(self, observations):
        """
        Args)
        observations            list(np.ndarray(shape=(2,2), dtype=np.float32))
        """
        assert len(observations) == self.dof()
        A = formulate_8point_matrix(observations)
        u, s, vh = np.linalg.svd(A)
        parameter = vh.T[:, -1].A1
        return parameter

    def dof(self):
        return 8

    def dim_parameter(self):
        return (9,)


class RANSAC:
    """
    RANSAC regression
    """

    def __init__(self, radius, iter):
        """
        Args)
        radius                  cost value radius
        iter                    number of iterations
        """
        self.radius = radius
        self.iter = iter
        self.rand = random.Random(0)  # fixed seed for reproducable results

    def solve(self, problem, population):
        """
        Solve a regression problem using RANSAC

        Args)
        problem                 A RegressionProblem instance
        population              np.ndarray(shape=(N,2,2), dtype=np.float32)))
        """
        consensus_parameter_pairs = []

        for _ in range(self.iter):
            samples = self.rand.sample(
                list(population), problem.dof())
            parameter = problem.estimate(samples)

            costs = map(partial(problem.cost, parameter), population)
            consensus = len(list(filter(lambda c: c < self.radius, costs)))
            consensus_parameter_pairs.append((consensus, parameter))

        best = sorted(consensus_parameter_pairs, key=lambda p: p[0])[-1]
        print(
            'RANSAC: selected the candidate with {} consensus'.format(best[0]))
        return best[1]


def estimate_fundamental_matrix(pair_point):
    """
    Estimate fundamental matrix from keypoints on an image to that of 
    another image

    Args)
    pair_point                  np.ndarray(shape=(N,2,2), dtype=np.float32)

    Returns)
    Fundamental matrix. np.ndarray(shape=(3,3), dtype=np.float)
    """
    ransac = RANSAC(options['ransac_radius'], options['ransac_iter'])
    problem = CameraExtrinsicProblem()
    population = pair_point

    parameter = ransac.solve(problem, population)
    F = get_fundamental_matrix(parameter)
    return F


def main_fundamental():
    """
    Estimate fundamental matrix from 

    Task 1-4
    """
    print('--------------------------------------------------------------------')
    print('Task 1-4')
    print('--------------------------------------------------------------------')
    pairs_image = get_all_scene_images(options['root_stereo_sample'])
    pairs_point = get_all_keypoint_matches(options['root_stereo_sample'])

    for i, ((image_left, image_right), pair_point) \
            in enumerate(zip(pairs_image, pairs_point)):
        F = estimate_fundamental_matrix(pair_point)
        path = os.path.join(
            options['root_stereo_sample'], 'scene{}_F.npy'.format(i))
        print('F for scene {} ({}) = '.format(i, path))
        print(F)
        np.save(path, F)


def get_essential_matrix(K, F):
    """
    Calculate essential matrix from a intrinsic matrix K and 
    a fundamental matrix F
    """
    return K.T @ F @ K


def main_essential():
    """
    Calculate essential matrix

    Task 1-5
    """
    print('--------------------------------------------------------------------')
    print('Task 1-5')
    print('--------------------------------------------------------------------')
    Fs = get_all_F(options['root_stereo_sample'])
    K = np.load(os.path.join(options['root_checker_sample'], 'K.npy'))

    for i, F in enumerate(Fs):
        E = get_essential_matrix(K, F)
        path = os.path.join(
            options['root_stereo_sample'], 'scene{}_E.npy'.format(i))
        print('E for scene{} ({})= '.format(i, path))
        print(E)
        np.save(path, E)


def decompose_essential_matrix(E):
    """
    Decompose essential matrix into rotation R and translation t

    returns 4 possible combination of transformations

    Returns)
    list((np.ndarray(shape=(3,3), dtype=np.float32), 
          np.ndarray(shape=(3,), dtype=np.float32)))
    """
    u, s, vh = np.linalg.svd(E)
    print('s=')
    print(s)
    u = np.matrix(u)
    vh = np.matrix(vh)
    W = np.matrix([[0.0, -1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0]])

    Rs = [np.array(u @ W @ vh), np.array(u @ W.T @ vh)]
    ts = [-u[:, 2].A1, u[:, 2].A1]

    return list(product(Rs, ts))


def main_decomposition():
    """
    Decompose essential matrix into rotation matrix R and translation t
    """
    print('--------------------------------------------------------------------')
    print('Task 1-6')
    print('--------------------------------------------------------------------')
    Es = get_all_E(options['root_stereo_sample'])
    for i, E in enumerate(Es):
        path = os.path.join(
            options['root_stereo_sample'], 'scene{}_pose.npz'.format(i))
        ms = decompose_essential_matrix(E)
        assert len(ms) == 4
        print('poses for scene {} ({})='.format(i, path))
        for j, (R, t) in enumerate(ms):
            print('option {} R:'.format(j))
            print(R)
            print('option {} t:'.format(j))
            print(t)
        np.savez(path,
                 R0=ms[0][0], t0=ms[0][1],
                 R1=ms[1][0], t1=ms[1][1],
                 R2=ms[2][0], t2=ms[2][1],
                 R3=ms[3][0], t3=ms[3][1])


"""
-------------------------------------------------------------------------------
Stereo matching
-------------------------------------------------------------------------------
"""


def rectify_images(im1, im2, K, distcoeffs, R, t):
    """
    Calculate rectified images
    """
    R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(
        K, distcoeffs, K, distcoeffs, (WIDTH_IMAGE, HEIGHT_IMAGE), R, t)

    map1, map2 = cv2.initUndistortRectifyMap(
        K, distcoeffs, R1, K, (WIDTH_IMAGE, HEIGHT_IMAGE), cv2.CV_32FC2)
    rectified_left = cv2.remap(im1, map1, map2, cv2.INTER_LINEAR)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, distcoeffs, R2, K, (WIDTH_IMAGE, HEIGHT_IMAGE), cv2.CV_32FC2)
    rectified_right = cv2.remap(im2, map1, map2, cv2.INTER_LINEAR)

    return rectified_left, rectified_right

def draw_horizontal_lines(im):
    """
    Draw red horizontal lines on image for visualization
    """
    im = im.copy()
    NUM_LINES = 20
    interval = HEIGHT_IMAGE // NUM_LINES
    for i in range(NUM_LINES):
        cv2.line(im, (0, interval*i), (WIDTH_IMAGE, interval*i), (255, 255, 255))
    return im

def visualize_rectified_image_pair(im1, im2):
    """
    Create a merged image with horizontal lines on them
    """
    return np.hstack(
        (draw_horizontal_lines(im1), 
         draw_horizontal_lines(im2)))
        

def main_rectify():
    prepare_root_inter(options['root_rectify_inter'])
    K = np.load(os.path.join(options['root_checker_sample'], 'K.npy'))
    distcoeffs = np.load(os.path.join(
        options['root_checker_sample'], 'dist.npy'))
    pair_images = get_all_scene_images(options['root_stereo_sample'])
    ms = get_all_pose(options['root_stereo_sample'])
    for i, ((image_left, image_right), m) in enumerate(zip(pair_images, ms)):
        Ts = [
            (m['R0'], m['t0']),
            (m['R1'], m['t1']),
            (m['R2'], m['t2']),
            (m['R3'], m['t3'])]
        for j, (R, t) in enumerate(Ts):
            rectified_left, rectified_right = \
                rectify_images(image_left, image_right, K, distcoeffs, R, t)
            path = os.path.join(
                options['root_rectify_inter'], 'scene{}_option{}.jpg'.format(i, j))
            print('Rectified images are stored in ({})'.format(path))
            cv2.imwrite(
                path, visualize_rectified_image_pair(rectified_left, rectified_right))


"""
-------------------------------------------------------------------------------
main
-------------------------------------------------------------------------------
"""

if __name__ == '__main__':
    # main_camera_intrinsic()
    # main_feature_matching()     # Task 1-3
    # main_fundamental()          # Task 1-4
    # main_essential()            # Task 1-5
    # main_decomposition()        # Task 1-6
    main_rectify()              # Task 2-1
