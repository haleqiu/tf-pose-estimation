
model = "mobilenet_thin"
image = "/data/drone_filming/images/forest_person_far_0.jpg"
to_write_path = "human_pose.txt"

import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import os, glob
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='tensorflow open pose')
parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
parser.add_argument("--base", type=str, default='/data/datasets', help="dataset location")
parser.add_argument("--folder", type=str, default='image_0', help="the name of the folder")
args = parser.parse_args(); print(args)

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def write_txt(to_write_path, width, height, humans):
    ## writing the human pose into the txt file
    to_write = []
    print(len(humans))
    for human in humans:
        item = []
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():# not detected hhh
                item+=[0,0,0]
                continue

            body_part = human.body_parts[i]
            center = [int(body_part.x * width + 0.5), int(body_part.y * height + 0.5), body_part.score]
            item+=center
        to_write.append(item)
    print(to_write_path)
    np.savetxt(to_write_path,np.array(to_write))


def tfpos_save(e, w, h, to_write_path, to_read, to_write_image):

    # estimate human poses from a single image !
    image = common.read_imgfile(to_read, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % to_read)
        sys.exit(-1)

    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (to_read, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    try:
        fig = plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.savefig(to_write_image)
        write_txt(to_write_path, w, h, humans)

    except Exception as e:
        logger.warning('matplitlib error, %s' % e)
        #cv2.imshow('result', image)

base_dir = args.base
read_folder = os.path.join(base_dir, args.folder)
to_write_folder = os.path.join(base_dir, args.folder+"_openpose")
to_write_image_folder = os.path.join(base_dir, args.folder+"_openpose_images")

if not os.path.isdir(to_write_folder):
    os.mkdir(to_write_folder)
if not os.path.isdir(to_write_image_folder):
    os.mkdir(to_write_image_folder)

image_names = glob.glob(read_folder+"/*")
e = TfPoseEstimator(get_graph_path(model), target_size=(640, 360))
for name in image_names:
    print(name)
    if not name[-3:] == "png":
        continue
    idx = os.path.basename(name).split("_")[-1][:-4]
    to_write_path = os.path.join(to_write_folder,idx+".txt")
    to_write_image = os.path.join(to_write_image_folder,idx+".png")


    tfpos_save(e, 640, 360, to_write_path, name, to_write_image)
