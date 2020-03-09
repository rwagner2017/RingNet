import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from psbody.mesh import Mesh

from config_test import get_config
from demo import preprocess_image
from run_RingNet import RingNet_inference


class FaceDataset(Dataset):
    """Face dataset."""

    def __init__(self, img_root_dir, img_size):
        self.img_root_dir = img_root_dir
        self.img_size = img_size

        self.img_paths = glob.glob(os.path.join(img_root_dir, '*'))
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_paths[idx]
        cropped_img, proc_param, img = preprocess_image(img_path, self.img_size)

        img_filename = os.path.basename(img_path)
        img_name = os.path.splitext(img_filename)[0]

        return {'image': cropped_img, 'name': img_name, 'proc_param': proc_param}


# # Helper function to show a batch
# def show_face_batch(images_batch):
#     batch_size = len(images_batch)
#     im_size = images_batch.size(2)
#     grid_border_size = 2
#
#     grid = utils.make_grid(images_batch)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))
#     plt.title('Batch from dataloader')


def main(config):
    transformed_dataset = FaceDataset(config.in_folder, config.img_size)

    # Dataloader
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=False, num_workers=4)

    template_mesh = Mesh(filename='./flame_model/FLAME_sample.ply')

    sess = tf.Session()
    model = RingNet_inference(config, sess=sess)

    inference_times = []
    start = time.time()
    for i_batch, sample_batched in enumerate(dataloader):

        n_images = len(sample_batched['image'])
        # print(i_batch, n_images)

        for idx in range(n_images):
            img = sample_batched['image'][idx]
            img_name = sample_batched['name'][idx]

            inf_start = time.time()
            vertices, flame_parameters = model.predict(np.expand_dims(img, axis=0), get_parameters=True)
            inf_end = time.time()
            duration = inf_end - inf_start
            inference_times.append(duration)

            if config.save_obj_file:
                if not os.path.exists(config.out_folder + '/mesh'):
                    os.mkdir(config.out_folder + '/mesh')
                mesh = Mesh(v=vertices[0], f=template_mesh.f)
                mesh.write_obj(config.out_folder + '/mesh/' +  '{}.obj'.format(img_name))

            if config.save_flame_parameters:
                if not os.path.exists(config.out_folder + '/params'):
                    os.mkdir(config.out_folder + '/params')
                flame_parameters_ = {'cam':  flame_parameters[0][:3], 'pose': flame_parameters[0][3:3+config.pose_params], 'shape': flame_parameters[0][3+config.pose_params:3+config.pose_params+config.shape_params],
                 'expression': flame_parameters[0][3+config.pose_params+config.shape_params:]}
                np.save(config.out_folder + '/params/' + '{}.npy'.format(img_name), flame_parameters_)

    end = time.time()
    overall_duration = end - start

    mean_inference_time = np.mean(inference_times)
    print('mean_inference_time = {}'.format(mean_inference_time))

    n_images = len(transformed_dataset)
    throughput = n_images / overall_duration
    print('total images = {} duration {} throughput = {}/s'.format(n_images, overall_duration, throughput))



if __name__ == '__main__':
    config = get_config()

    if not os.path.exists(config.out_folder):
        os.makedirs(config.out_folder)

    main(config)