import torch
import numpy as np
from skimage.transform import EssentialMatrixTransform
from skimage.measure import ransac
from xfeat.accelerated_features.modules.xfeat import XFeat
import cv2

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
        assert K.shape == (3, 3), "Camera matrix K must be 3x3"
        self.xfeat = XFeat()
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(K)

    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, :2]

    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):
        img_tensor = torch.from_numpy(img).permute(
            2, 0, 1).unsqueeze(0).float() / 255.0
        output = self.xfeat.detectAndCompute(img_tensor, top_k=3000)[0]
        kps, des = output['keypoints'], output['descriptors']

        if des is not None and not isinstance(des, np.ndarray):
            des = des.cpu().numpy()

        if not len(kps):
            return [], None

        kps = [(float(kp[0]), float(kp[1])) for kp in kps]
        ret = []

        if self.last:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = sorted(
                bf.match(des, self.last['des']), key=lambda x: x.distance)

            for m in matches:
                kp1 = kps[m.queryIdx]
                kp2 = self.last['kps'][m.trainIdx]
                ret.append((kp1, kp2))

        Rt = None
        if len(ret) >= 8:
            ret = np.array(ret)
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            try:
                model, inliers = ransac(
                    (ret[:, 0], ret[:, 1]),
                    EssentialMatrixTransform,
                    min_samples=8,
                    residual_threshold=0.005,
                    max_trials=200
                )
                if inliers.any():
                    ret, Rt = ret[inliers], extractRt(model.params)
            except Exception:
                pass

        self.last = {'kps': kps, 'des': des}
        return ret, Rt
