import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def convert_ringnet(model_dir, checkpoint_name):
    print('Tensorflow version {}'.format(tf.__version__))

    load_path = os.path.join(model_dir, checkpoint_name)
    print('Loading saved model from {}'.format(load_path + '.meta'))
    saver = tf.train.import_meta_graph(load_path + '.meta')
    graph = tf.get_default_graph()

    print('Restoring checkpoint %s..' % load_path)
    sess = tf.Session()
    saver.restore(sess, load_path)

    # import ipdb; ipdb.set_trace()

    ops = graph.get_operations()
    tensors = tf.contrib.graph_editor.get_tensors(graph)

    # file_writer = tf.summary.FileWriter('logs', sess.graph)

    input_tensor = tensors[0]
    input_tensor_shape = input_tensor.shape

    input_size = (None, input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3])
    placeholder = tf.contrib.graph_editor.make_placeholder_from_dtype_and_shape(tf.float32, shape=input_size)
    images_pl = tf.placeholder(tf.float32, shape=input_size, name='input_images_new')

    # c_ = tf.contrib.graph_editor.graph_replace(input_tensor, {placeholder: images_pl})

    sgv = tf.contrib.graph_editor.make_view_from_scope(ops, graph)

    pass


if __name__ == '__main__':
    model_dir = r'./model'
    checkpoint_name = r'ring_6_68641'
    convert_ringnet(model_dir, checkpoint_name)