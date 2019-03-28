import numpy as np
import tensorflow as tf
import os
from __future__ import division, print_function, unicode_literals
from io import open
# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import sys
import tarfile
from six.moves import urllib

FLOWERS_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_PATH = os.path.join("datasets", "flowers")

def fetch_flowers(url=FLOWERS_URL, path=FLOWERS_PATH):
    if os.path.exists(FLOWERS_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "flower_photos.tgz")
    urllib.request.urlretrieve(url, tgz_path)
    flowers_tgz = tarfile.open(tgz_path)
    flowers_tgz.extractall(path=path)
    flowers_tgz.close()
    os.remove(tgz_path)

fetch_flowers()

flowers_root_path = os.path.join(FLOWERS_PATH, "flower_photos")
flower_classes = sorted([dirname for dirname in os.listdir(flowers_root_path)
                  if os.path.isdir(os.path.join(flowers_root_path, dirname))])
flower_classes

from collections import defaultdict

image_paths = defaultdict(list)
for flower_class in flower_classes:
    image_dir = os.path.join(flowers_root_path,flower_class)
    for file_path in os.listdir(image_dir):
        if(file_path.endswith('.jpg')):
            image_paths[flower_class].append(os.path.join(image_dir,file_path))

for paths in image_paths.values():
    paths.sort()

import matplotlib.image as mping

n_examples_per_class = 2
channels = 3

for flower_class in flower_classes:
    print('class:',flower_class)
    plt.figure(figsize=(10,5))
    for index,example_image_path in enumerate(image_paths[flower_class][:n_examples_per_class]):
        example_image = mping.imread(example_image_path)[:,:,:channels]
        plt.subplot(100+n_examples_per_class*10+index+1)
        plt.title('{}x{}'.format(example_image.shape[1],example_image.shape[0]))
        plt.imshow(example_image)
        plt.axis('off')
    plt.show()

from scipy.misc import imresize

def prepare_image(image,target_width=299,target_height=299,max_zoom=0.2):
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width/height
    target_image_ratio = target_width/target_height
    crop_vertically = image_ratio<target_image_ratio
    crop_width = width if crop_vertically else int(height*target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height

    resize_factor = np.random.rand()*max_zoom+1.0
    crop_width = int(crop_width/resize_factor)
    crop_height = int(crop_height/resize_factor)

    x0 = np.random.randint(0,width-crop_width)
    y0 = np.random.randint(0,height-crop_height)
    x1 = x0+crop_width
    y1 = y0+crop_height

    image = image[y0:y1,x0:x1]

    if np.random.rand()<0.5:
        image = np.fliplr(image)

    image = imresize(image,(target_width,target_height))

    return image.astype(np.float32)/255

plt.figure(figsize=(6, 8))
plt.imshow(example_image)
plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
plt.axis("off")
plt.show()

prepared_image = prepare_image(example_image)
plt.imshow(prepared_image)
plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
plt.axis("off")
plt.show()

rows, cols = 2, 3

plt.figure(figsize=(14, 8))
for row in range(rows):
    for col in range(cols):
        prepared_image = prepare_image(example_image)
        plt.subplot(rows, cols, row * cols + col + 1)
        plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
        plt.imshow(prepared_image)
        plt.axis("off")
plt.show()


from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

reset_graph()

height, width = 299,299

X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
training = tf.placeholder_with_default(False, shape=[])
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)

inception_saver = tf.train.Saver()

logits.op.inputs[0]

logits.op.inputs[0].op.inputs[0]

logits.op.inputs[0].op.inputs[0].op.inputs[0]

end_points

end_points["PreLogits"]



prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])

n_outputs = len(flower_classes)
with tf.name_scope('new_output_layer'):
    flower_logits = tf.layers.dense(prelogits,n_outputs,name='flower_logits')
    Y_proba = tf.nn.softmax(flower_logits,name='Y_proba')

y = tf.placeholder(tf.int32, shape=[None])
#############冻结前面的层，只训练后面的


with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flower_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="flower_logits")
    training_op = optimizer.minimize(loss, var_list=flower_vars)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(flower_logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.name_scope('init_and_save'):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

[v.name for v in flower_vars]

flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}
flower_class_ids
flower_paths_and_class = []
for flower_class,paths in image_paths.items():
    for path in paths:
        flower_paths_and_class.append((path,flower_class_ids[flower_class]))

#########划分训练集和验证集
test_ratio = 0.2
train_size = int(len(flower_paths_and_class)*(1-test_ratio))

np.random.shuffle(flower_paths_and_class)

flower_paths_and_classes_train = flower_paths_and_class[:train_size]
flower_paths_and_classes_test = flower_paths_and_class[train_size:]
#########c产生batch的数据

from random import sample

def prepare_batch(flower_paths_and_classes,batch_size):
    batch_paths_and_classes = sample(flower_paths_and_classes,batch_size)
    images = [mping.imread(path)[:,:,:channels] for path,labels in batch_paths_and_classes]
    prepare_images = [prepare_image(image) for image in images]
    X_batch = 2*np.stack(prepare_images)-1####incepton 需要输入-1到1
    y_batch = np.array([labels for path,labels in batch_paths_and_classes],dtype=np.int32)
    return X_batch,y_batch

X_batch,y_batch = prepare_batch(flower_paths_and_class,4)
X_batch.shape

X_test, y_test = prepare_batch(flower_paths_and_classes_test, batch_size=len(flower_paths_and_classes_test))

n_epochs = 10
batch_size = 40
n_iterations_per_epoch = len(flower_paths_and_classes_train)//batch_size


TF_MODELS_URL = "http://download.tensorflow.org/models"
INCEPTION_V3_URL = TF_MODELS_URL + "/inception_v3_2016_08_28.tar.gz"
INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")


with tf.Session() as sess:
    init.run()
    inception_saver.restore(sess,INCEPTION_V3_CHECKPOINT_PATH)
    for epoch in range(n_epochs):
        print('epoch',epoch,end='')
        for iteration in range(n_iterations_per_epoch):
            print('.',end='')
            X_batch,y_batch = prepare_batch(flower_paths_and_classes_train,batch_size)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch,training:True})
        acc_batch = accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        print("  Last batch accuracy:", acc_batch)

        save_path = saver.save(sess,'./my_flower_model')

n_test_batches = 10
X_test_batches = np.array_split(X_test, n_test_batches)
y_test_batches = np.array_split(y_test, n_test_batches)

with tf.Session() as sess:
    saver.restore(sess, "./my_flowers_model")

    print("Computing final accuracy on the test set (this will take a while)...")
    acc_test = np.mean([
        accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch})
        for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])
    print("Test accuracy:", acc_test)
