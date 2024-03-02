import cv2
import os
import numpy as np
from study.DetectHV.detect_thumbfinger_point import DetectThumbFingerPoint
from study.DetectHV.get_tip_finger import GetTipFinger


class DetectM1M5Line:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.color_image = self.image.copy()
        self.detect_thumb_finger_point = DetectThumbFingerPoint(image_path)
        self.get_tip_finger = GetTipFinger(image_path)

    def canny_image(self):
        smooth_image = cv2.GaussianBlur(self.image, (3, 3), 0)
        edges = cv2.Canny(smooth_image, threshold1=40, threshold2=200)

        # クロージング処理（収縮→膨張）を適用する
        close_kernel = np.ones((9, 9), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

        return closed_edges

    # 入力のy座標位置は一回目の白いピクセルの検出に使用して、入力のy座標位置から20ピクセル分上がったところを二回目の白いピクセル検出の際にマークする
    def detect_points(self, image, left_point1, right_point1):
        height, width = image.shape[:2]
        count = 0
        left_point1 = left_point1
        right_point1 = right_point1
        left_point2 = left_point1 - 10
        right_point2 = right_point1 - 10
        detect_right1, detect_left1, detect_right2, detect_left2 = (
            None,
            None,
            None,
            None,
        )

        ##### 左足の特徴点を検出 #####
        for i in range(100, width - 100, 1):
            if image[left_point1, i] == 0 and image[left_point1, i + 1] == 255:
                detect_left1 = (i, left_point1)
                break

        count = 0
        for i in range(100, width - 100, 1):
            if (
                image[left_point2, i] == 0
                and image[left_point2, i + 1] == 255
                and count == 0
            ):
                count = 1

            elif (
                image[left_point2, i] == 255
                and image[left_point2, i + 1] == 0
                and count == 1
            ):
                count = 2

            elif (
                image[left_point2, i] == 0
                and image[left_point2, i + 1] == 255
                and count == 2
            ):
                count = 3

            elif (
                image[left_point2, i] == 255
                and image[left_point2, i + 1] == 0
                and count == 3
            ):
                detect_left2 = (i, left_point2)
                break

        ###### 右足の特徴点を検出 #####
        for i in range(width - 100, 100, -1):
            if image[right_point1, i] == 0 and image[right_point1, i - 1] == 255:
                detect_right1 = (i, right_point1)
                break

        count = 0
        for i in range(width - 100, 100, -1):
            if (
                image[right_point2, i] == 0
                and image[right_point2, i - 1] == 255
                and count == 0
            ):
                count = 1

            elif (
                image[right_point2, i] == 255
                and image[left_point2, i - 1] == 0
                and count == 1
            ):
                count = 2

            elif (
                image[right_point2, i] == 0
                and image[right_point2, i - 1] == 255
                and count == 2
            ):
                count = 3

            elif (
                image[right_point2, i] == 255
                and image[right_point2, i - 1] == 0
                and count == 3
            ):
                detect_right2 = (i, right_point2)
                break

        return detect_right1, detect_right2, detect_left1, detect_left2

    ###################　直線を描画する　###################

    def calculate_midpoint(self, point1, point2):
        return ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

    def draw_line(self, point1, point2):
        cv2.line(self.color_image, point1, point2, (0, 0, 255), 10)

    def draw_midpoint_lines(self, points, line_length=300):
        # 4つの点に対して中点を計算し、直線を描画
        for i in range(0, len(points), 4):
            midpoint1 = self.calculate_midpoint(points[i], points[i + 1])
            midpoint2 = self.calculate_midpoint(points[i + 2], points[i + 3])
            self.draw_line(midpoint1, midpoint2)

            # 直線を長く引くために中点から適切な距離だけ離した座標を計算
            dx = midpoint2[0] - midpoint1[0]
            dy = midpoint2[1] - midpoint1[1]
            length = np.sqrt(dx**2 + dy**2)
            dx = int(dx / length * line_length)
            dy = int(dy / length * line_length)
            start_point = (midpoint1[0] - dx, midpoint1[1] - dy)
            end_point = (midpoint2[0] + dx, midpoint2[1] + dy)
            self.draw_line(start_point, end_point)

            # 中点の位置に円を描画
            cv2.circle(self.color_image, midpoint1, 10, (255, 0, 0), -1)
            cv2.circle(self.color_image, midpoint2, 10, (255, 0, 0), -1)

        return (midpoint1, midpoint2)

    ################### main ###################

    def main(self):
        canny = self.canny_image()
        (
            _,
            left_thumb_finger,
            right_thumb_finger,
        ) = self.detect_thumb_finger_point.process_image()
        tip_point_R = self.get_tip_finger.find_first_white_pixel(direction="right")
        tip_point_L = self.get_tip_finger.find_first_white_pixel(direction="left")

        length_R = right_thumb_finger[1] - tip_point_R[1]
        length_L = left_thumb_finger[1] - tip_point_L[1]

        left_1 = int(left_thumb_finger[1] - length_L * 0.025)
        right_1 = int(right_thumb_finger[1] - length_R * 0.025)

        left_2 = int(left_thumb_finger[1] - length_L * 0.12)
        right_2 = int(right_thumb_finger[1] - length_R * 0.12)

        right1, right2, left1, left2 = self.detect_points(canny, left_1, right_1)
        right1_2, right2_2, left1_2, left2_2 = self.detect_points(
            canny, left_2, right_2
        )

        right_mid_point = self.draw_midpoint_lines((right1, right2, right1_2, right2_2))
        left_mid_point = self.draw_midpoint_lines((left1, left2, left1_2, left2_2))

        cv2.circle(
            self.color_image, (int(right1[0]), int(right1[1])), 5, (0, 255, 0), -1
        )
        cv2.circle(
            self.color_image, (int(right2[0]), int(right2[1])), 5, (0, 255, 0), -1
        )
        cv2.circle(self.color_image, (int(left1[0]), int(left1[1])), 5, (0, 255, 0), -1)
        cv2.circle(self.color_image, (int(left2[0]), int(left2[1])), 5, (0, 255, 0), -1)
        cv2.circle(
            self.color_image, (int(right1_2[0]), int(right1_2[1])), 5, (0, 255, 0), -1
        )
        cv2.circle(
            self.color_image, (int(right2_2[0]), int(right2_2[1])), 5, (0, 255, 0), -1
        )
        cv2.circle(
            self.color_image, (int(left1_2[0]), int(left1_2[1])), 5, (0, 255, 0), -1
        )
        cv2.circle(
            self.color_image, (int(left2_2[0]), int(left2_2[1])), 5, (0, 255, 0), -1
        )

        return self.color_image, canny, right_mid_point, left_mid_point


if __name__ == "__main__":
    input_image_folder = "src/images"
    output_image_folder1 = "assets/M1M5_line"
    output_image_folder2 = "assets/canny_M1M5"
    if not os.path.exists(output_image_folder1):
        os.mkdir(output_image_folder1)
    image_files = os.listdir(input_image_folder)

    if not os.path.exists(output_image_folder2):
        os.mkdir(output_image_folder2)
    image_files = os.listdir(input_image_folder)

    for image_file in image_files:
        input_image = os.path.join(input_image_folder, image_file)
        print(input_image)
        output_image_point = os.path.join(output_image_folder1, image_file)
        output_image = os.path.join(output_image_folder2, image_file)

        test = DetectM1M5Line(input_image)
        image, canny, _, _ = test.main()
        cv2.imwrite(output_image_point, image)
        cv2.imwrite(output_image, canny)
