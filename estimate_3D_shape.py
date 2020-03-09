import sys
import os
import glob
from absl import flags
import numpy as np
import skimage.io as io
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model
import time

from util import renderer as vis_util
from util import image as img_util
from config_test import get_config
from run_RingNet import RingNet_inference
from demo import preprocess_image, visualize, visualize_single_row


def main(config):
    print('Tensorflow version {}'.format(tf.__version__))

    print("Input Dir: <{}>".format(config.in_folder))
    print("Output Dir: <{}>".format(config.out_folder))

    img_paths = glob.glob(os.path.join(config.in_folder, '*'))

    if (config.save_viz or config.save_obj_file):
        template_mesh = Mesh(filename='./flame_model/FLAME_sample.ply')

    sess = tf.Session()
    model = RingNet_inference(config, sess=sess)

    pre_process_times = []
    inference_times = []
    start = time.time()
    for img_path in img_paths:
        pre_start = time.time()
        input_img, proc_param, img = preprocess_image(img_path, config.img_size)
        pre_end = time.time()
        duration = pre_end - pre_start
        pre_process_times.append(duration)

        inf_start = time.time()
        vertices, flame_parameters = model.predict(np.expand_dims(input_img, axis=0), get_parameters=True)
        inf_end = time.time()
        duration = inf_end - inf_start
        inference_times.append(duration)

        if config.save_viz:
            if not os.path.exists(config.out_folder + '/images'):
                os.mkdir(config.out_folder + '/images')

            cams = flame_parameters[0][:3]
            renderer = vis_util.SMPLRenderer(faces=template_mesh.f)
            # visualize(img, proc_param, vertices[0], cams, renderer, img_name=config.out_folder + '/images/' + img_path.split('/')[-1][:-4])
            visualize_single_row(img, proc_param, vertices[0], cams, renderer, img_name=config.out_folder + '/images/' + img_path.split('/')[-1][:-4])

        if config.save_obj_file:
            if not os.path.exists(config.out_folder + '/mesh'):
                os.mkdir(config.out_folder + '/mesh')
            mesh = Mesh(v=vertices[0], f=template_mesh.f)
            mesh.write_obj(config.out_folder + '/mesh/' + img_path.split('/')[-1][:-4] + '.obj')

        if config.save_flame_parameters:
            if not os.path.exists(config.out_folder + '/params'):
                os.mkdir(config.out_folder + '/params')
            flame_parameters_ = {'cam':  flame_parameters[0][:3], 'pose': flame_parameters[0][3:3+config.pose_params], 'shape': flame_parameters[0][3+config.pose_params:3+config.pose_params+config.shape_params],
             'expression': flame_parameters[0][3+config.pose_params+config.shape_params:]}
            np.save(config.out_folder + '/params/' + img_path.split('/')[-1][:-4] + '.npy', flame_parameters_)

        if config.neutralize_expression:
            from util.using_flame_parameters import make_prdicted_mesh_neutral
            if not os.path.exists(config.out_folder + '/neutral_mesh'):
                os.mkdir(config.out_folder + '/neutral_mesh')
            neutral_mesh = make_prdicted_mesh_neutral(config.out_folder + '/params/' + img_path.split('/')[-1][:-4] + '.npy', config.flame_model_path)
            neutral_mesh.write_obj(config.out_folder + '/neutral_mesh/' + img_path.split('/')[-1][:-4] + '.obj')
    end = time.time()
    overall_duration = end - start

    mean_pre_process_time = np.mean(pre_process_times)
    mean_inference_time = np.mean(inference_times)

    print('mean_pre_process_time = {}'.format(mean_pre_process_time))
    print('mean_inference_time = {}'.format(mean_inference_time))

    n_images = len(img_paths)
    throughput = n_images / np.sum(pre_process_times) + np.sum(inference_times)
    print('total images = {} throughput = {}/s'.format(n_images, throughput))

    throughput = n_images / overall_duration
    print('total images = {} duration {} throughput = {}/s'.format(n_images, overall_duration, throughput))

if __name__ == '__main__':
    config = get_config()

    if not os.path.exists(config.out_folder):
        os.makedirs(config.out_folder)

    main(config)