from detected_lengthA import GetLengthA
from Preprocessing.preprocessing import PreprocessingBinary, PreprocessingCanny
import cv2
import os


class SegmentPoints:
    def __init__(self, image_path):
        self.image_path = image_path

    def detect_point_HV(self, image, L1, L2, R1, R2, progress):
        height, width = image.shape[:2]
        half_width = int(width / 2)

        L1_x, L2_x, R1_x, R2_x = None, None, None, None

        for i in range(half_width, 150, -1):
            if image[L1, i] == 0 and image[L1, i - 1] == 255:
                L1_x = int(i - progress)
                break

        for i in range(half_width, 150, -1):
            if image[L2, i] == 0 and image[L2, i - 1] == 255:
                L2_x = int(i - progress)
                break

        for i in range(half_width, width - 150, 1):
            if image[R1, i] == 0 and image[R1, i + 1] == 255:
                R1_x = int(i + progress)
                break

        for i in range(half_width, width - 150, 1):
            if image[R2, i] == 0 and image[R2, i + 1] == 255:
                R2_x = int(i + progress)
                break
        return (L1_x, L1), (L2_x, L2), (R1_x, R1), (R2_x, R2)

    def detect_point_M1M5(self, image, L1, R1, progress):
        height, width = image.shape[:2]
        half_width = int(width / 2)
        (
            L1_x,
            R1_x,
        ) = (
            None,
            None,
        )

        for i in range(0, half_width, 1):
            if image[L1, i] == 0 and image[L1, i + 1] == 255:
                L1_x = int(i + progress)
                break

        for i in range(width - 1, half_width, -1):
            if image[R1, i] == 0 and image[R1, i - 1] == 255:
                R1_x = int(i - progress)
                break

        return (L1_x, L1), (R1_x, R1)

    def detect_point_background(self, image, L1, R1, L2, R2, progress):
        height, width = image.shape[:2]
        half_width = int(width / 2)
        L1_x, R1_x, L2_x, R2_x = None, None, None, None

        for i in range(150, width - 1, 1):
            if image[L1, i] == 0 and image[L1, i + 1] == 255:
                L1_x = int(i + progress)
                break

        for i in range(width - 150, half_width, -1):
            if image[R1, i] == 0 and image[R1, i - 1] == 255:
                R1_x = int(i - progress)
                break

        for i in range(150, width - 1, 1):
            if image[L2, i] == 0 and image[L1, i + 1] == 255:
                L2_x = int(i + (progress - 200))
                break

        for i in range(width - 150, half_width, -1):
            if image[R2, i] == 0 and image[R1, i - 1] == 255:
                R2_x = int(i - (progress - 200))
                break

        return (L1_x, L1), (L2_x, L2), (R1_x, R1), (R2_x, R2)

    # HV,M1M5両方の処理をかく
    def main(self):
        get_length = GetLengthA(image_path=self.image_path)
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        print(
            "############################################################  HV  ############################################################"
        )
        # A*0.3＋親指の先端の長さ(HV一つ目のプロンプトのy)
        L1, R1 = get_length.get_length(0.3)
        # A*0.7＋親指の先端の長さ(HV二つ目のプロンプトのy)
        L2, R2 = get_length.get_length(0.7)

        preprocess_canny = PreprocessingCanny(
            type="HV", image=image, threshold1=50, threshold2=180
        )
        canny_HV = preprocess_canny.process_image()

        # HV角検出に使用する2つの対象骨プロンプトの座標
        L1_HV, L2_HV, R1_HV, R2_HV = self.detect_point_HV(
            image=canny_HV, L1=L1, L2=L2, R1=R1, R2=R2, progress=45
        )

        # エッジ画像に余分な線がある関係上プロンプトがずれる可能性があるのでずれていると判断した場合はbinaryに対し処理を行う
        if abs(L1_HV[0] - L2_HV[0]) > 20 or abs(R1_HV[0] - R2_HV[0]) > 20:
            print("HV binaryに入った")
            preprocess_binary = PreprocessingBinary(
                type="HV", image=image, threshold1=150, threshold2=255
            )
            binary_HV = preprocess_binary.process_image()
            # HV角検出に使用する2つの対象骨プロンプトの座標
            L1_HV, L2_HV, R1_HV, R2_HV = self.detect_point_HV(
                image=binary_HV, L1=L1, L2=L2, R1=R1, R2=R2, progress=45
            )

        print(
            "############################################################  M1M5  ############################################################"
        )
        # A*0.9＋親指の先端の長さ(M1M5のプロンプトのy)
        L1, R1 = get_length.get_length(0.9)

        preprocess_canny = PreprocessingCanny(
            type="M1M5", image=image, threshold1=50, threshold2=180
        )
        canny_M1M5 = preprocess_canny.process_image()
        # HV角検出に使用する2つの対象骨プロンプトの座標
        L1_M1M5, R1_M1M5 = self.detect_point_M1M5(
            image=canny_M1M5, L1=L1, R1=R1, progress=30
        )

        print(
            "############################################################  background  ############################################################"
        )
        # A*1.4＋親指の先端の長さ(背景のプロンプトのy)
        L1, R1 = get_length.get_length(1.4)

        # A*0.9＋親指の先端の長さ(背景のプロンプトのy)
        L2, R2 = get_length.get_length(0.9)

        # HV角検出に使用する2つの対象骨プロンプトの座標
        preprocess_binary = PreprocessingBinary(
            type="background", image=image, threshold1=120, threshold2=255
        )
        binary_background = preprocess_binary.process_image()
        (
            L1_background,
            L2_background,
            R1_background,
            R2_background,
        ) = self.detect_point_background(
            image=binary_background, L1=L1, R1=R1, L2=L2, R2=R2, progress=250
        )

        return (
            (L1_HV, L2_HV, R1_HV, R2_HV),
            (L1_M1M5, R1_M1M5),
            (L1_background, L2_background, R1_background, R2_background),
        )


if __name__ == "__main__":
    input_folder = "/home/kubota/study_backup1216/segment-anything/src/images"
    output_folder = "/home/kubota/study_backup1216/segment-anything/assets/prompts"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".png"):
            input_image_path = os.path.join(input_folder, file_name)
            print(input_image_path)
            output_image_path = os.path.join(output_folder, f"prompt_{file_name}")
            color_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
            segment_points = SegmentPoints(image_path=input_image_path)
            HV_prompt, M1M5_prompt, background_prompt = segment_points.main()

            cv2.circle(color_image, HV_prompt[0], 10, (0, 0, 255), -1)
            cv2.circle(color_image, HV_prompt[1], 10, (0, 0, 255), -1)
            cv2.circle(color_image, HV_prompt[2], 10, (0, 0, 255), -1)
            cv2.circle(color_image, HV_prompt[3], 10, (0, 0, 255), -1)
            cv2.circle(color_image, M1M5_prompt[0], 10, (0, 255, 0), -1)
            cv2.circle(color_image, M1M5_prompt[1], 10, (0, 255, 0), -1)
            cv2.circle(color_image, background_prompt[0], 10, (255, 0, 0), -1)
            cv2.circle(color_image, background_prompt[1], 10, (255, 0, 0), -1)
            cv2.circle(color_image, background_prompt[2], 10, (255, 0, 0), -1)
            cv2.circle(color_image, background_prompt[3], 10, (255, 0, 0), -1)

            cv2.imwrite(output_image_path, color_image)
