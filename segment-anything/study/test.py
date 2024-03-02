import os
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import detected_lengthA


class Test_M1M2:
    # 対象の閾値以下の輝度値の部分を赤くマスクしてその重心を取得
    def make_red_mask(self, image, threshold):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # 輝度値が閾値以下のピクセルを赤色にマスクする
        mask = image < threshold
        rgb_image[mask] = [0, 0, 255]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8)
        )

        # 面積が大きい順にラベルをソート
        sorted_labels = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1

        # 上位2つのラベルに対して処理
        for label in sorted_labels[: min(2, len(sorted_labels))]:
            centroid = centroids[label]
            cv2.circle(
                rgb_image, (int(centroid[0]), int(centroid[1])), 10, (255, 0, 0), -1
            )

        # 重心が描画された赤いマスク画像と重心の座標を返す
        return rgb_image, sorted(
            [
                (int(centroids[label][0]), int(centroids[label][1]))
                for label in sorted_labels[: min(2, len(sorted_labels))]
            ],
            key=lambda x: x[0],  # x座標に基づいてソート
        )

    # 指定したy座標から画像を真横に切ってその輝度値を表示
    def make_hist(self, image, file_name, y_coord):
        row = image[y_coord, :]

        # x座標の値を生成（0から画像の幅-1まで）
        x_coords = range(row.shape[0])

        plt.figure(figsize=(12, 4))
        plt.plot(x_coords, row, color="blue")
        plt.title(f"Intensity Profile at y={y_coord}")
        plt.xlabel("x Coordinate")
        plt.ylabel("Pixel Value")
        plt.savefig(f"{file_name}_hist.png")
        plt.close()


if __name__ == "__main__":
    image_dir = "segment-anything/src/images"
    output_dir = "/home/kubota/study_backup1216/segment-anything/assets/test"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    test_M1M2 = Test_M1M2()

    for file_name in os.listdir(image_dir):
        input_image_path = os.path.join(image_dir, file_name)
        output_image_path = os.path.join(output_dir, file_name)

        original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        denoised_image = cv2.medianBlur(original_image, 7)

        get_lengthA = detected_lengthA.GetLengthA(image_path=input_image_path)

        # 両足の先端検出
        (start_x_right, start_y_right) = get_lengthA.detect_point_a(direction="right")
        (start_x_left, start_y_left) = get_lengthA.detect_point_a(direction="left")

        redmask, centroid = test_M1M2.make_red_mask(original_image, threshold=30)

        centroid_left, centroid_right = centroid[0], centroid[1]

        # 赤いマスクの重心と足の先端からおおよその足の全長を計算
        full_length_left, full_length_right = (
            centroid_left[1] - start_y_left,
            centroid_right[1] - start_y_right,
        )

        line_left, line_right = (
            int(start_y_left + full_length_left * 0.4),
            int(start_y_right + full_length_right * 0.4),
        )

        cv2.circle(redmask, (start_x_left, start_y_left), 10, (255, 0, 0), -1)
        cv2.circle(redmask, (start_x_right, start_y_right), 10, (255, 0, 0), -1)
        cv2.line(redmask, (0, line_left), (10000, line_left), (0, 255, 0), 10)
        cv2.line(redmask, (0, line_right), (10000, line_right), (0, 255, 0), 10)

        cv2.imwrite(output_image_path, redmask)
        print(line_left, line_right)

        # 画像のヒストグラムを作る
        test_M1M2.make_hist(
            original_image,
            os.path.join(output_dir, "original_" + file_name),
            y_coord=line_left,
        )

        test_M1M2.make_hist(
            denoised_image,
            os.path.join(output_dir, "denoised_" + file_name),
            y_coord=line_left,
        )
