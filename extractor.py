import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


class Extractor:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bg = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.last = None
        self.camPos = [0, 0, 0]
        self.cam_xyz = []
        self.lm_xyz = []
        self.scale = 2
        self.all_landmarks = []
        self.all_camera_positions = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()

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
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)
            matched_keypoints = matched_keypoints[inliers]

        self.last = {'kps': kps, 'des': des}
        return matched_keypoints

    def pointClouds(self, W, H, points1, points2):
        W, H = W // 2, H // 2

        # focal length
        fov = 60 * (math.pi / 180)
        f_w = W / math.tan(fov / 2)
        f_h = H / math.tan(fov / 2)

        # camera matrix parameters
        K = np.array([[f_w, 0, W],
                     [0, f_h, H],
                     [0, 0, 1]])

        E, mask = cv2.findFundamentalMat(np.float32(points2),
                                         np.float32(points1),
                                         cv2.FM_8POINT)

        # rotation and translation
        points, R, t, mask = cv2.recoverPose(E, np.float32(points2),
                                             np.float32(points1), K, 500)
        R = np.asmatrix(R).I

        # update camera position
        new_cam_pos = [self.camPos[0] + t[0][0],
                       self.camPos[1] + t[1][0],
                       self.camPos[2] + t[2][0]]
        self.all_camera_positions.append(new_cam_pos)

        # camera matrix
        C = np.hstack((R, t))
        P = np.asmatrix(K) * np.asmatrix(C)

        frame_landmarks = []
        for i in range(len(points2)):
            x_i = np.asmatrix([points2[i][0], points2[i][1], 1]).T
            X_i = np.asmatrix(P).I * x_i
            landmark = [X_i[0][0] * self.scale + self.camPos[0],
                        X_i[1][0] * self.scale + self.camPos[1],
                        X_i[2][0] * self.scale + self.camPos[2]]
            frame_landmarks.append(landmark)

        self.all_landmarks.extend(frame_landmarks)
        self.camPos = new_cam_pos

    def update_plot(self):
        self.ax.cla()

        landmarks = np.array(self.all_landmarks)
        camera_positions = np.array(self.all_camera_positions)

        if len(landmarks) > 0:
            self.ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2],
                            c='b', s=1, label='Landmarks')

        if len(camera_positions) > 0:
            self.ax.scatter(camera_positions[:, 0], camera_positions[:, 1],
                            camera_positions[:, 2], c='r', s=50,
                            label='Camera Positions')
            self.ax.plot(camera_positions[:, 0], camera_positions[:, 1],
                         camera_positions[:, 2], 'r-', linewidth=1)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)
