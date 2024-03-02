import os
import cv2
import numpy as np
import torch
import os
import time
from segment_points import SegmentPoints

import sys

new_path = "/home/kubota/study_backup1216/segment-anything"

if new_path not in sys.path:
    sys.path.append(new_path)

for path in sys.path:
    print(path)

from segment_anything import sam_model_registry, SamPredictor


class SegmentAnythingProcess:
    def __init__(self, model_checkpoint_path, model_type, device):
        sam = sam_model_registry[model_type](checkpoint=model_checkpoint_path)
        sam = sam.to(device=device)
        self.predictor = SamPredictor(sam)

    # promptと画像を入力
    def process_image(
        self,
        image_path,
        type,
        prompt_HV,
        prompt_M1M5,
        prompt_background,
    ):
        image = cv2.imread(image_path)
        self.predictor.set_image(image)
        mask_HV_L, mask_HV_R, mask_M1M5_L, mask_M1M5_R = None, None, None, None

        ################################################# HV角の検出 #################################################
        if type == "HV":
            print(
                "############################## HV Segment Anything ##############################"
            )

            # 左足の検出
            prompt_coords = np.vstack(
                (prompt_HV[0], prompt_HV[1], prompt_background[0], prompt_background[1])
            )
            print(prompt_coords)
            input_label = np.array([1, 1, 0, 0])
            masks, scores, logits = self.predictor.predict(
                point_coords=prompt_coords,
                point_labels=input_label,
                multimask_output=True,
            )
            mask_HV_L = self.get_mask_with_max_score(masks, scores)
            mask_HV_L = (mask_HV_L * 255).astype(np.uint8)
            print(mask_HV_L)

            # 右足の検出
            prompt_coords = np.vstack(
                (prompt_HV[2], prompt_HV[3], prompt_background[2], prompt_background[3])
            )
            print(prompt_coords)
            input_label = np.array([1, 1, 0, 0])
            masks, scores, logits = self.predictor.predict(
                point_coords=prompt_coords,
                point_labels=input_label,
                multimask_output=True,
            )
            mask_HV_R = self.get_mask_with_max_score(masks, scores)
            mask_HV_R = (mask_HV_R * 255).astype(np.uint8)

            return mask_HV_L, mask_HV_R

        ################################################# M1M5角の検出 #################################################

        elif type == "M1M5":
            print(
                "############################## M1M5 Segment Anything ##############################"
            )

            # 左足の検出
            prompt_coords = np.vstack(
                (prompt_M1M5[0], prompt_background[0], prompt_background[1])
            )
            print(prompt_coords)
            input_label = np.array([1, 0, 0])
            masks, scores, logits = self.predictor.predict(
                point_coords=prompt_coords,
                point_labels=input_label,
                multimask_output=False,
            )
            mask_M1M5_L = self.get_mask_with_max_score(masks, scores)
            mask_M1M5_L = (mask_M1M5_L * 255).astype(np.uint8)

            # 右足の検出
            prompt_coords = np.vstack(
                (prompt_M1M5[1], prompt_background[2], prompt_background[3])
            )
            print(prompt_coords)
            input_label = np.array([1, 0, 0])
            masks, scores, logits = self.predictor.predict(
                point_coords=prompt_coords,
                point_labels=input_label,
                multimask_output=False,
            )
            mask_M1M5_R = self.get_mask_with_max_score(masks, scores)
            mask_M1M5_R = (mask_M1M5_R * 255).astype(np.uint8)

            return mask_M1M5_L, mask_M1M5_R

    # scoreが最大のmaskを取得する
    def get_mask_with_max_score(self, masks, scores):
        scores = scores.tolist()  # numpy.ndarray をリストに変換
        max_score = max(scores)  # スコアの最大値を取得
        max_index = scores.index(max_score)  # 最大スコアのインデックスを取得
        max_mask = masks[max_index]  # 最大スコアに対応するマスクを取得
        return max_mask


if __name__ == "__main__":
    input_folder = "/home/kubota/study_backup1216/segment-anything/src/images"
    output_folderHV = "/home/kubota/study_backup1216/segment-anything/assets/MASKHV"
    output_folderM1M5 = "/home/kubota/study_backup1216/segment-anything/assets/MASKM1M5"

    if not os.path.exists(output_folderHV):
        os.makedirs(output_folderHV)
    if not os.path.exists(output_folderM1M5):
        os.makedirs(output_folderM1M5)

    sam_checkpoint = (
        "/home/kubota/study_backup1216/segment-anything/study/sam_vit_h_4b8939.pth"
    )
    model_type = "vit_h"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    segment_anything_process = SegmentAnythingProcess(
        sam_checkpoint, model_type, device
    )

    image_files = os.listdir(input_folder)
    for file_name in image_files:
        if file_name.endswith(".png"):
            input_image_path = os.path.join(input_folder, file_name)

            print(
                f"##########################  {input_image_path}  ##########################"
            )

            segment_points = SegmentPoints(image_path=input_image_path)
            HV_prompts, M1M5_prompts, background_prompts = segment_points.main()

            mask_HV_L, mask_HV_R = segment_anything_process.process_image(
                image_path=input_image_path,
                type="HV",
                prompt_HV=HV_prompts,
                prompt_M1M5=M1M5_prompts,
                prompt_background=background_prompts,
            )

            mask_M1M5_L, mask_M1M5_R = segment_anything_process.process_image(
                image_path=input_image_path,
                type="M1M5",
                prompt_HV=HV_prompts,
                prompt_M1M5=M1M5_prompts,
                prompt_background=background_prompts,
            )

            # 生成されたマスクを保存
            output_path_HV_L = os.path.join(output_folderHV, f"mask_HV_L_{file_name}")
            output_path_HV_R = os.path.join(output_folderHV, f"mask_HV_R_{file_name}")
            output_path_M1M5_L = os.path.join(
                output_folderM1M5, f"mask_M1M5_L_{file_name}"
            )
            output_path_M1M5_R = os.path.join(
                output_folderM1M5, f"mask_M1M5_R_{file_name}"
            )

            cv2.imwrite(output_path_HV_L, mask_HV_L)
            cv2.imwrite(output_path_HV_R, mask_HV_R)
            cv2.imwrite(output_path_M1M5_L, mask_M1M5_L)
            cv2.imwrite(output_path_M1M5_R, mask_M1M5_R)
