from skimage.transform import EssentialMatrixTransform
from skimage.measure import ransac
import cv2
import numpy as np
np.set_printoptions(suppress=True)


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def extractRt(E):
    U, d, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    R = np.dot(U, np.dot(W, Vt))
    if np.sum(R.diagonal()) < 0:
        R = np.dot(U, np.dot(W.T, Vt))

    t = U[:, 2]
    Rt = np.concatenate([R, t.reshape(3, 1)], axis=1)
    return Rt


class Extractor(object):
    def __init__(self, K):
        assert K.shape == (3, 3), "camera matrix K must be 3x3"
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(K)

    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, :2]

    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(
            np.uint8), 3000, qualityLevel=0.02, minDistance=1)
        if feats is None or len(feats) < 8:
            print("Insufficient features")
            return [], None

        # extraction and matching
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        ret = []
        if self.last:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    kp1, kp2 = kps[m.queryIdx].pt, self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        Rt = None
        if len(ret) >= 8:
            ret = np.array(ret)
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            try:
                model, inliers = ransac((ret[:, 0], ret[:, 1]), EssentialMatrixTransform,
                                        min_samples=8, residual_threshold=0.005, max_trials=200)
                if inliers.any():
                    ret = ret[inliers]
                    Rt = extractRt(model.params)
                else:
                    print("no inliers found")
            except Exception as e:
                print(f"extraction error: {e}")

        self.last = {'kps': kps, 'des': des}
        return ret, Rt
