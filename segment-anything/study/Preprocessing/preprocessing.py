import cv2
import os
import numpy as np


# 四角形角度を検出
def angle_between(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# 四角形を画像から除去
def remove_squares(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            # 角度が90度に近いものを判断してそれを除去する
            angles = []
            for i in range(4):
                angle = angle_between(
                    approx[i % 4][0], approx[(i + 1) % 4][0], approx[(i + 2) % 4][0]
                )
                angles.append(angle)
            if all(85 <= angle <= 95 for angle in angles):
                squares.append(cnt)

    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, squares, -1, (255, 255, 255), -1)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


# 画像の直線を除去
def remove_long_lines(image, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(
        image,
        1,
        np.pi / 180,
        100,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return image


# エッジ画像の前処理
class PreprocessingCanny:
    def __init__(self, type, image, threshold1, threshold2):
        self.image = image
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.type = type

    def process_image(self):
        smoothed_image = cv2.GaussianBlur(self.image, (3, 3), 0)

        edges = cv2.Canny(smoothed_image, self.threshold1, self.threshold2)
        dilate_kernel = np.ones((9, 9), np.uint8)

        if self.type == "HV":
            # HV
            dilated_edges = cv2.dilate(edges, dilate_kernel, iterations=1)

        elif self.type == "M1M5":
            # M1M5
            dilated_edges = cv2.dilate(edges, dilate_kernel, iterations=1)

        final_image = remove_squares(dilated_edges)
        final_image = remove_long_lines(
            final_image, min_line_length=2000, max_line_gap=0
        )
        return final_image


# 二値化画像の前処理
class PreprocessingBinary:
    def __init__(self, type, image, threshold1, threshold2):
        self.image = image
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.type = type

    def process_image(self):
        if self.type == "HV":
            ret, binary = cv2.threshold(
                self.image, self.threshold1, self.threshold2, cv2.THRESH_BINARY_INV
            )
        if self.type == "background":
            ret, binary = cv2.threshold(
                self.image, self.threshold1, self.threshold2, cv2.THRESH_BINARY_INV
            )

        final_image = remove_squares(binary)
        # final_image = remove_long_lines(
        #     final_image, min_line_length=1000, max_line_gap=0
        # )
        return final_image


if __name__ == "__main__":
    # エッジ画像の前処理
    # HV
    input_folder = "src/images"
    output_folder = "assets/canny_HV"
    threshold1 = 50
    threshold2 = 180

    # M1M5
    # input_folder = "src/images"
    # output_folder = "assets/canny_M1M5"
    # threshold1 = 45
    # threshold2 = 200

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [file for file in os.listdir(input_folder) if file.endswith(".png")]

    for image_file in image_files:
        file_path = os.path.join(input_folder, image_file)

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        preprocessing = PreprocessingCanny(
            type="HV", image=image, threshold1=threshold1, threshold2=threshold2
        )
        final_image = preprocessing.process_image()

        file_name = os.path.basename(image_file)
        output_file_name = f"HV_{file_name}"
        output_image_path = os.path.join(output_folder, output_file_name)
        cv2.imwrite(output_image_path, final_image)
        print(f"ファイルを変換して保存しました: {output_image_path}")

    # 二値化画像の前処理
    input_folder = "src/images"
    output_folder = "assets/binary"
    threshold1 = 120
    threshold2 = 255

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [file for file in os.listdir(input_folder) if file.endswith(".png")]

    for image_file in image_files:
        file_path = os.path.join(input_folder, image_file)

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        preprocessing = PreprocessingBinary(
            type="HV", image=image, threshold1=threshold1, threshold2=threshold2
        )
        final_image = preprocessing.process_image()

        file_name = os.path.basename(image_file)
        output_file_name = f"binary_{file_name}"
        output_image_path = os.path.join(output_folder, output_file_name)
        cv2.imwrite(output_image_path, final_image)
        print(f"ファイルを変換して保存しました: {output_image_path}")
