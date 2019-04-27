import cv2 as cv
import numpy as np
from utils import extract_area

class LineAlarm:

    def __init__(self, src_points, dst_dims):

        x, y, w, h = dst_dims
        q0 = [x    , y]
        q1 = [x + w, y]
        q2 = [x    , y + h]
        q3 = [x + w, y + h]

        # Points to do the homography
        self.src_points = src_points
        self.dst_points = np.array([q0, q1, q2, q3])

        # Rectangles that define the limit to trigger the alarms
        self.upper_left_rect =  ((int(x - 2 * w + w / 2), 0), (int(x - w / 2)            , y))
        self.upper_right_rect = ((int(x + w + w / 2)    , 0), (int(x + w + 2 * w - w / 2), y))
        self.lower_left_rect =  ((int(x - 2 * w + w / 2), y), (int(x - w / 2)            , y + h))
        self.lower_right_rect = ((int(x + w + w / 2)    , y), (int(x + w + 2*w - w/2)    , y + h))

        # Transformation matrix of the homography
        self.h = None

        # Arrays to keep an history of the movement of the plates along the
        # bands
        self.history_left = None
        self.history_right = None

        # Factor to avoid None types due to the limit of the image
        self.offset = 3

        # Points (frame indices) where the alarms are triggered
        self.alarms = []

    def init(self, frame):
        # Calculate the transformation Matrix
        self.h, status = cv.findHomography(self.src_points, self.dst_points)

        # Processed image and movement points
        i, l, r = self.__process_frame(frame)
        self.history_left, self.history_right = np.array(l), np.array(r)

    def process_frame(self, frame):
        if self.h is None:
            raise ValueError('Homography matrix is not initialized')

        # Processed image and movement points
        i, l, r = self.__process_frame(frame)

        # Append the new points to history
        self.history_left = np.hstack((self.history_left, l))
        self.history_right = np.hstack((self.history_right, r))

        return i

    def warp(self, frame):
        return cv.warpPerspective(frame, self.h, (frame.shape[1], frame.shape[0]))

    def draw_rects(self, warpFrame):
        cv.rectangle(warpFrame, self.upper_left_rect[0], self.upper_left_rect[1], (255, 0, 0), 1, 1)
        cv.rectangle(warpFrame, self.upper_right_rect[0], self.upper_right_rect[1], (255, 0, 0), 1, 1)
        cv.rectangle(warpFrame, self.lower_left_rect[0], self.lower_left_rect[1], (0, 0, 255), 1, 1)
        cv.rectangle(warpFrame, self.lower_right_rect[0], self.lower_right_rect[1], (0, 0, 255), 1, 1)

        return warpFrame

    def calculate_alarms(self):

        upper_left = np.argwhere(self.history_left >= self.upper_left_rect[1][1] - self.offset).ravel()
        upper_right = np.argwhere(self.history_right >= self.upper_right_rect[1][1] - self.offset).ravel()
        lower_left = np.argwhere(self.history_left >= self.lower_left_rect[1][1] - self.offset).ravel()
        lower_right = np.argwhere(self.history_right >= self.lower_right_rect[1][1] - self.offset).ravel()

        upper_left = self.__def_limit(upper_left)
        upper_right = self.__def_limit(upper_right)
        lower_left = self.__def_limit(lower_left)
        lower_right = self.__def_limit(lower_right)

        self.alarms = [upper_left, upper_right, lower_left, lower_right]

    def __process_frame(self, frame):
        # Frame in zenital view
        warp = self.warp(frame)

        # Limits to threshold the image in the Light channel
        o_l, o_h = 0, 20

        # Limits to threshold the image in RGB channels to detect orange color
        # r_l, r_h, g_l, g_h, b_l, b_h = 230, 255, 115, 170, 50, 85

        # Arrays for thresheld images
        th = []
        gr = []

        # Merge the upper and lower rectangles
        box_left = (self.upper_left_rect[0], self.lower_left_rect[1])
        box_right = (self.upper_right_rect[0], self.lower_right_rect[1])

        # In every rectangle...
        for idx, box in enumerate([box_left, box_right]):
            # Box of interest
            boi = extract_area(warp,
                box[0][0],
                box[0][1],
                box[1][0] - box[0][0],
                box[1][1] - box[0][1])
            # HLS
            boi = cv.cvtColor(boi, cv.COLOR_BGR2HLS)
            # Lightness channel
            boi = boi[:,:,1]
            # Pixels within the threshold limits
            bools = (boi > o_l) & (boi < o_h)
            # Thresheld box
            binary = np.zeros_like(boi)

            # b = boi[:,:,0]
            # g = boi[:,:,1]
            # r = boi[:,:,2]
            # bools = (r > r_l) & (r < r_h) & (g > g_l) & (g < g_h) & (b > b_l) & (b < b_h)
            # binary = np.zeros_like(boi[:,:,0])

            # Valid pixels as white
            binary[bools == True] = 1

            # Dilate and erode for better analysis
            kernel1 = np.ones((5,5), np.uint8)
            binary = cv.dilate(binary, kernel1, iterations=1)
            kernel2 = np.ones((3,3), np.uint8)
            binary = cv.erode(binary, kernel2, iterations=1)

            # First white pixel
            max_ = int(np.argmax(binary==1) / binary.shape[1])
            if idx == 0:
                max_left = max_
            elif idx == 1:
                max_right = max_

            # White out pixels below first white pixel
            binary[max_:,:] = 1
            th.append(binary)

            # Grayscale image
            box = extract_area(warp,
                box[0][0],
                box[0][1],
                box[1][0] - box[0][0],
                box[1][1] - box[0][1])
            box = cv.cvtColor(box, cv.COLOR_BGR2RGB)
            gr.append(box)

        # Left and right thresheld bands
        th_image = np.hstack((th[0], th[1]))*255
        th_image = cv.resize(th_image,  (th_image.shape[1]*3, th_image.shape[0]*3))

        # left and right grayscale bands
        gr_image = np.hstack((gr[0], gr[1]))*255

        # Limit lines drawn in the grayscale bands
        gr_image = cv.cvtColor(gr_image, cv.COLOR_BGR2GRAY)
        upper = self.upper_left_rect[1][1] - self.offset
        lower = self.lower_left_rect[1][1] - self.offset
        gr_image[upper,:] = 0
        gr_image[lower,:] = 0
        gr_image = cv.resize(gr_image,  (gr_image.shape[1]*3, gr_image.shape[0]*3))

        # Merge thresheld and grayscale images
        gr_image = np.hstack((th_image, gr_image))

        return gr_image.T, max_left, max_right

    def __def_limit(self, indices):
        if indices.shape[0] > 0:
            return indices[0]
        else:
            return None
