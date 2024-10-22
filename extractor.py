import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform


class Extractor:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bg = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.last = None

    # turn [x, y] -> [x, y, 1]
    def add_ones(self, x):
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

    def extract(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(
            gray, maxCorners=3000, qualityLevel=0.01, minDistance=2)

        if features is None:
            return []

        kps = []
        for feat in features:
            feat = cv2.KeyPoint(x=feat[0][0], y=feat[0][1], size=10)
            kps.append(feat)

        kps, des = self.orb.compute(img, kps)

        # print(des.shape)
        matched_keypoints = []

        if self.last is not None:
            matches = self.bg.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    matched_keypoints.append((kp1, kp2))

        if len(matched_keypoints) > 0:
            matched_keypoints = np.array(matched_keypoints)
            model, inliers = ransac((matched_keypoints[:, 0], matched_keypoints[:, 1]),
                                    EssentialMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)
            matched_keypoints = matched_keypoints[inliers]

        self.last = {'kps': kps, 'des': des}
        return matched_keypoints
