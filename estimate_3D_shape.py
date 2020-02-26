import sys
import os
import glob
from absl import flags
import numpy as np
import skimage.io as io
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model

from util import renderer as vis_util
from util import image as img_util
from config_test import get_config
from run_RingNet import RingNet_inference
from demo import preprocess_image, visualize


def main(config):
    print("Input Dir: <{}>".format(config.in_folder))
    print("Output Dir: <{}>".format(config.out_folder))

    img_paths = glob.glob(os.path.join(config.in_folder, '*'))

    for img_path in img_paths:
        sess = tf.Session()
        model = RingNet_inference(config, sess=sess)
        input_img, proc_param, img = preprocess_image(img_path, config.img_size)
        vertices, flame_parameters = model.predict(np.expand_dims(input_img, axis=0), get_parameters=True)

        if config.save_viz:
            if not os.path.exists(config.out_folder + '/images'):
                os.mkdir(config.out_folder + '/images')

            cams = flame_parameters[0][:3]
            template_mesh = Mesh(filename='./flame_model/FLAME_sample.ply')
            renderer = vis_util.SMPLRenderer(faces=template_mesh.f)
            visualize(img, proc_param, vertices[0], cams, renderer, img_name=config.out_folder + '/images/' + img_path.split('/')[-1][:-4])

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


if __name__ == '__main__':
    config = get_config()

    if not os.path.exists(config.out_folder):
        os.makedirs(config.out_folder)

    main(config)