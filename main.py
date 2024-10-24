import time
import cv2
from display import Display
from extractor import Extractor
import numpy as np

W = 1920//2
H = 1080//2

disp = Display(W, H)
fe = Extractor()

window_name = 'a'


def process_frame(img):
    img = cv2.resize(img, (W, H))
    # keypoints = fe.extract(img)
    matches = fe.extract(img)
    print(f'matches: {len(matches)}')
    for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    disp.paint(img)


if __name__ == "__main__":
    cap = cv2.VideoCapture("test_countryroad.mp4")

    # tensor
    x = np.array([[1, 2]])
    d = fe.add_ones(x)
    print(d)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
