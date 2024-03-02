import numpy as np

import concurrent.futures
import os
import pydicom
import time
from PIL import Image


def dicom_to_png(dicom_path, output_folder):
    dcm_file = pydicom.dcmread(dicom_path)
    dcm_img = dcm_file.pixel_array

    # 最大値と最小値のピクセルを0に変更
    max_val = dcm_img.max()
    min_val = dcm_img.min()
    dcm_img[dcm_img == max_val] = 16383
    dcm_img[dcm_img == min_val] = 16383
    print("Adjusted Max and Min:", dcm_img.max(), dcm_img.min())

    wc = dcm_file.WindowCenter
    ww = dcm_file.WindowWidth
    ri = dcm_file.RescaleIntercept
    rs = dcm_file.RescaleSlope

    img = dcm_img * rs + ri
    vmax = wc + ww / 2
    vmin = wc - ww / 2
    img = 255 * (img - vmin) / (vmax - vmin)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    dicom_image = Image.fromarray(img)

    file_name = os.path.splitext(os.path.basename(dicom_path))[0] + ".png"
    output_path = os.path.join(output_folder, file_name)
    dicom_image.save(output_path)
    print(f"File saved: {output_path}")


if __name__ == "__main__":
    start = time.time()
    input_folder = "segment-anything/src/dicom"
    output_folder = "segment-anything/src/images"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dicom_files = [
        os.path.join(input_folder, file_name)
        for file_name in os.listdir(input_folder)
        if file_name.endswith(".dcm")
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda x: dicom_to_png(x, output_folder), dicom_files)
    end = time.time()
    print("time : ", end - start)
