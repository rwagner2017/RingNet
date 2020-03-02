import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def show_crop(img_path, bbox_path):
    bbox = np.load(bbox_path, allow_pickle=True, encoding='latin1').item()
    top = bbox['top']
    right = bbox['right']
    bottom = bbox['bottom']
    left = bbox['left']

    img = cv.imread(img_path)
    face_crop = img[top:bottom, left:right, :]

    fig = plt.figure(1)
    plt.imshow(face_crop)

    height = bottom - top + 1
    width = right - left + 1

    scaled_bottom = int(bottom + 0.05 * height)
    scaled_left = int(left - .1 * width)
    scaled_right = int(right + .1 * width)
    scaled_top = int(top - 0.3 * height)

    scaled_face_crop = img[scaled_top:scaled_bottom, scaled_left:scaled_right, :]

    fig = plt.figure(2)
    plt.imshow(scaled_face_crop)

    cropped_img_path = os.path.join('/mnt/data/head_study/NoW_results', os.path.basename(img_path))
    cv.imwrite(cropped_img_path, scaled_face_crop)

    pass


if __name__ == '__main__':
    img_path = r'/mnt/data/head_study/NoW_Dataset/final_release_version/iphone_pictures/FaMoS_180424_03335_TA/multiview_expressions/IMG_0054.jpg'
    bbox_path = r'/mnt/data/head_study/NoW_Dataset/final_release_version/detected_face/FaMoS_180424_03335_TA/multiview_expressions/IMG_0054.npy'
    show_crop(img_path, bbox_path)

    plt.show()