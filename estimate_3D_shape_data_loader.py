import sys
import os
import glob
from absl import flags
import numpy as np
import skimage.io as io
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model
import time

from util import renderer as vis_util
from util import image as img_util
from config_test import get_config
from run_RingNet import RingNet_inference
from demo import preprocess_image, visualize

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

  # Normalize image to [-1, 1]
  img = tf.math.subtract(img, -0.5)
  img = tf.math.multiply(img, 2)

  return img


def process_path(file_path):
  # label = get_label(file_path)
  label = 0
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)

  return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return ds

def prepare_for_testing(ds, cache=True):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return ds


def preprocess_image_224(img_path):
    return preprocess_image(img_path, 224)


import time
default_timeit_steps = 1000

def timeit(ds, steps=default_timeit_steps):
  start = time.time()
  it = iter(ds)
  for i in range(steps):
    batch = next(it)
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))


def main(config):
    print('Tensorflow version {}'.format(tf.__version__))

    print("Input Dir: <{}>".format(config.in_folder))
    print("Output Dir: <{}>".format(config.out_folder))

    img_paths = glob.glob(os.path.join(config.in_folder, '*/*'))
    n_images = len(img_paths)

    for img_path in img_paths:
        print(img_path)

    list_ds = tf.data.Dataset.list_files(os.path.join(config.in_folder, '*/*'))
    # for f in list_ds.take(5):
    #     print(f.numpy())

    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_ds = prepare_for_testing(labeled_ds)
    print(test_ds.element_spec)

    sess = tf.Session()
    model = RingNet_inference(config, sess=sess)

    # iterator = test_ds.make_one_shot_iterator()
    iterator = tf.data.Iterator.from_structure(test_ds.output_types,
                                               test_ds.output_shapes)

    testing_init_op = iterator.make_initializer(test_ds)
    model.sess.run(testing_init_op)

    pre_process_times = []
    inference_times = []


    for idx in range(n_images):
        start = time.time()
        next_element = iterator.get_next()
        input_img = next_element[0].eval(session=model.sess)[0]
        end = time.time()
        duration = end - start
        pre_process_times.append(duration)


        start = time.time()
        vertices, flame_parameters = model.predict(np.expand_dims(input_img, axis=0), get_parameters=True)
        end = time.time()
        duration = end - start
        inference_times.append(duration)

        # print(flame_parameters)

    pass

    mean_pre_process_time = np.mean(pre_process_times)
    mean_inference_time = np.mean(inference_times)

    print('mean_pre_process_time = {}'.format(mean_pre_process_time))
    print('mean_inference_time = {}'.format(mean_inference_time))

    n_images = len(img_paths)
    throughput = n_images / np.sum(pre_process_times) + np.sum(inference_times)
    print('total images = {} throughput = {}/s'.format(n_images, throughput))

    # for img_path in img_paths:
    #     sess = tf.Session()
    #     model = RingNet_inference(config, sess=sess)
    #     input_img, proc_param, img = preprocess_image(img_path, config.img_size)
    #
    #     # vertices, flame_parameters = model.predict(np.expand_dims(input_img, axis=0), get_parameters=True)
    #     # print('vertices shape = {}'.format(vertices.shape))
    #     # print('flame_parameters shape = {}'.format(flame_parameters.shape))
    #
    #     results = model.predict_dict([input_img, input_img])
    #     print(results)
    #     vertices = results['vertices']
    #     flame_parameters = results['parameters']
    #     print('dict vertices shape = {}'.format(vertices.shape))
    #     print('dict flame_parameters shape = {}'.format(flame_parameters.shape))
    #
    #     if config.save_viz:
    #         if not os.path.exists(config.out_folder + '/images'):
    #             os.mkdir(config.out_folder + '/images')
    #
    #         cams = flame_parameters[0][:3]
    #         template_mesh = Mesh(filename='./flame_model/FLAME_sample.ply')
    #         renderer = vis_util.SMPLRenderer(faces=template_mesh.f)
    #         visualize(img, proc_param, vertices[0], cams, renderer, img_name=config.out_folder + '/images/' + img_path.split('/')[-1][:-4])
    #
    #     if config.save_obj_file:
    #         if not os.path.exists(config.out_folder + '/mesh'):
    #             os.mkdir(config.out_folder + '/mesh')
    #         mesh = Mesh(v=vertices[0], f=template_mesh.f)
    #         mesh.write_obj(config.out_folder + '/mesh/' + img_path.split('/')[-1][:-4] + '.obj')
    #
    #     if config.save_flame_parameters:
    #         if not os.path.exists(config.out_folder + '/params'):
    #             os.mkdir(config.out_folder + '/params')
    #         flame_parameters_ = {'cam':  flame_parameters[0][:3], 'pose': flame_parameters[0][3:3+config.pose_params], 'shape': flame_parameters[0][3+config.pose_params:3+config.pose_params+config.shape_params],
    #          'expression': flame_parameters[0][3+config.pose_params+config.shape_params:]}
    #         np.save(config.out_folder + '/params/' + img_path.split('/')[-1][:-4] + '.npy', flame_parameters_)
    #
    #     if config.neutralize_expression:
    #         from util.using_flame_parameters import make_prdicted_mesh_neutral
    #         if not os.path.exists(config.out_folder + '/neutral_mesh'):
    #             os.mkdir(config.out_folder + '/neutral_mesh')
    #         neutral_mesh = make_prdicted_mesh_neutral(config.out_folder + '/params/' + img_path.split('/')[-1][:-4] + '.npy', config.flame_model_path)
    #         neutral_mesh.write_obj(config.out_folder + '/neutral_mesh/' + img_path.split('/')[-1][:-4] + '.obj')


if __name__ == '__main__':
    config = get_config()

    if not os.path.exists(config.out_folder):
        os.makedirs(config.out_folder)

    main(config)
