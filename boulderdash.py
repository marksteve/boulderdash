import cv2
import numpy as np


def detect_circles(frame):
    img = cv2.medianBlur(frame, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=200,
                               param2=100, minRadius=0, maxRadius=0)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
    return img


def color_boundaries(color):
    ci = ['red', 'green', 'blue'].index(color.lower())
    lower = [0] * 3
    lower[ci] = 64
    upper = [64] * 3
    upper[ci] = 255
    return lower, upper


def detect_color(frame, color):
    lower, upper = color_boundaries(color)
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(frame, lower, upper)
    return cv2.bitwise_and(frame, frame, mask=mask)


cv2.ocl.setUseOpenCL(False)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)


def remove_background(frame):
    return fgbg.apply(frame)


def main():
    cv2.namedWindow('boulderdash')
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        ret, frame = vc.read()
    else:
        ret = False

    while ret:
        # output = detect_circles(frame)
        # output = detect_color(frame, 'green')
        output = remove_background(frame)

        cv2.imshow('boulderdash', output)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

        ret, frame = vc.read()

    cv2.destroyWindow('boulderdash')


if __name__ == '__main__':
    main()

