from matplotlib import pyplot as plt
from cv2 import cv2
import numpy as np
import fire
import json
import glob
import math
import os


reference_points = np.array([
    [0, 0],       # tl
    [1000, 0],    # tr
    [1000, 1000], # br
    [0, 1000],    # bl
])


def process_instance(im, im_info):
    points = np.array(im_info['canonical_board']['tl_tr_br_bl'])

    matrix, _ = cv2.findHomography(points, reference_points)

    warped_im = cv2.warpPerspective(im, matrix, (1000, 1000))
    gray_im = cv2.cvtColor(warped_im, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_im, cv2.HOUGH_GRADIENT, 1, 60, param1=400, param2=15, minRadius=30, maxRadius=40)

    checkers_count = {
        'top': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        'bottom': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(warped_im,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(warped_im,(i[0],i[1]),2,(0,0,255),3)

        x, y, _ = i

        if 480 < x < 530:
            continue

        if y < 400:
            spot = 'top'
        elif y > 600:
            spot = 'bottom'
        else:
            continue

        pip = math.floor(x/85)
        checkers_count[spot][pip] += 1

    plt.figure()
    plt.imshow(cv2.cvtColor(warped_im, cv2.COLOR_BGR2RGB))
    plt.show()

    return checkers_count, warped_im


def process_images(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for im_f in glob.glob(os.path.join(src_dir, '*.jpg')):
        info_f = im_f + '.info.json'
        
        im = cv2.imread(im_f)
        info = json.load(open(info_f, 'r'))

        checkers_count, warped_im = process_instance(im, info)
        im_name = os.path.split(im_f)[-1]

        cv2.imwrite(os.path.join(dst_dir, f'{im_name}.visual_feedback.jpg'), warped_im)
        json.dump(checkers_count, open(os.path.join(dst_dir, f'{im_name}.checkers.json'), 'w'))


if __name__ == "__main__":
    process_images('bgsamples', 'bgsamples_out')