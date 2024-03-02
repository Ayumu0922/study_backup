import numpy as np
import os
import cv2
from Preprocessing.preprocessing import PreprocessingCanny


class GetLengthA:
    def __init__(self, image_path):
        self.image_path = image_path

    # 親指の先端(a)を検出する
    def detect_point_a(self, direction):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        preprocessing_canny = PreprocessingCanny(
            type="HV", image=image, threshold1=50, threshold2=180
        )
        self.canny = preprocessing_canny.process_image()
        height, width = self.canny.shape[:2]
        start_x = 0
        start_y = 0
        white_pixel_found = False

        y_range = range(0, height)
        x_range = (
            range(width // 2, width)
            if direction == "right"
            else range(width // 2, 0, -1)
        )

        for y in y_range:
            for x in x_range:
                pixel_color = self.canny[y, x]
                if pixel_color == 255 and not white_pixel_found:
                    start_x = x
                    start_y = y
                    white_pixel_found = True
                    break
            if white_pixel_found:
                break
        return (start_x, start_y)

    # 足の内輪郭線を検出し、親指の付け根の点(b)を検出する
    def detect_point_b(self):
        # 足の内輪郭線を取得
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        canny = cv2.Canny(image, 60, 200)
        height, width = canny.shape[:2]
        inner = np.zeros((height, width), dtype=np.uint8)
        for y in range(0, height):
            find_inner = False
            for x in range(width // 2, width):
                if canny[y, x] == 255 and not find_inner:
                    find_inner = True
                    inner[y, x] = 255
                    break

        for y in range(0, height):
            find_inner = False
            for x in range(width // 2, 0, -1):
                if canny[y, x] == 255 and not find_inner:
                    find_inner = True
                    inner[y, x] = 255
                    break

        # shi-tomasiを使用して特徴点を検出
        corners = cv2.goodFeaturesToTrack(
            inner, 10, 0.01, 70, blockSize=40, useHarrisDetector=False, k=0.04
        )

        if corners is not None:
            half_height = image.shape[0] // 2

            # 画像の高さの半分よりも大きいy座標を２つ取得
            points = [i.ravel() for i in corners if i.ravel()[1] > half_height]

            # 特徴点が二つ以下なら何もしない
            if len(points) < 2:
                print("特徴点が検出できません")
                return image, None, None

            # 特徴点の中から左右の座標を選ぶ
            points_sorted = sorted(points, key=lambda x: x[1])
            min_points = points_sorted[:2]

            # ｙ座標の差
            y_diff = abs(min_points[0][1] - min_points[1][1])

            if y_diff > 100:
                # ｙ座標の差が100以上であればもう一度harrisを使用して検出しなおす
                corners = cv2.goodFeaturesToTrack(
                    inner,
                    10,
                    0.01,
                    70,
                    blockSize=40,
                    useHarrisDetector=True,
                    k=0.04,
                )

                points = [i.ravel() for i in corners if i.ravel()[1] > half_height]
                points_sorted = sorted(points, key=lambda x: x[1])
                min_points = points_sorted[:2]

            # min_pointsの中で画像を半分で分けたとき、半分より大きいものを右足の座標として、半分より小さいものを左足の座標として返す
            if len(min_points) == 2:
                half_width = image.shape[1] // 2

                left_points = [point for point in min_points if point[0] < half_width]
                right_points = [point for point in min_points if point[0] >= half_width]

                if len(left_points) == 1 and len(right_points) == 1:
                    left_thumb_finger = left_points[0]
                    right_thumb_finger = right_points[0]

                    return left_thumb_finger, right_thumb_finger

        print("特徴点が検出できません")
        return None, None

    # 画像に指定された始点から特定の長さだけ垂直方向に直線を描画する。
    def draw_vertical_line(
        self, img, start_point, length, color=(0, 0, 255), thickness=10
    ):
        end_point = (start_point[0], start_point[1] + length)
        cv2.line(img, start_point, end_point, color, thickness)
        return img

    # 長さを検出して返すメソッド(lengthにはスケールを設定する)
    def get_length(self, length):
        # point_a
        (start_x_right, start_y_right) = self.detect_point_a(direction="right")
        (start_x_left, start_y_left) = self.detect_point_a(direction="left")

        # point_b
        left_thumb_finger, right_thumb_finger = self.detect_point_b()

        # 今回使用する部分の直線を引く
        left_length_A = int(((left_thumb_finger[1]) - start_y_left) * length)
        right_length_A = int(((right_thumb_finger[1]) - start_y_right) * length)

        # 返り値は0からの長さを返す
        right_length = right_length_A + start_y_right
        left_length = left_length_A + start_y_left

        return left_length, right_length


if __name__ == "__main__":
    input_folder = "/home/kubota/study_backup1216/segment-anything/src/images"
    output_folder = (
        "/home/kubota/study_backup1216/segment-anything/assets/length_A * 1.4"
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".png"):
            input_image_path = os.path.join(input_folder, file_name)
            output_image_path = os.path.join(output_folder, f"inner_{file_name}")

            color_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

            get_lengthA = GetLengthA(input_image_path)
            # point_a
            (start_x_right, start_y_right) = get_lengthA.detect_point_a(
                direction="right"
            )
            (start_x_left, start_y_left) = get_lengthA.detect_point_a(direction="left")

            # point_b
            left_thumb_finger, right_thumb_finger = get_lengthA.detect_point_b()

            # 今回使用する部分の直線を引く
            left_length_A = int(((left_thumb_finger[1]) - start_y_left) * 1.4)
            right_length_A = int(((right_thumb_finger[1]) - start_y_right) * 1.4)

            image_with_line = get_lengthA.draw_vertical_line(
                color_image,
                start_point=(start_x_left, start_y_left),
                length=left_length_A,
            )

            image_with_line = get_lengthA.draw_vertical_line(
                color_image,
                start_point=(start_x_right, start_y_right),
                length=right_length_A,
            )

            cv2.imwrite(output_image_path, color_image)
