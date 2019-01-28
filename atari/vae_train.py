'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae.vae import ConvVAE, reset_graph

# Hyperparameters for ConvVAE
z_size=32
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5

# Parameters for training
NUM_EPOCH = 10
DATA_DIR = "record"
# IMAGE_DATA_DIR = "record_image"
IMAGE_DATA_DIR = "record_image_uncompressed"
if not os.path.exists(IMAGE_DATA_DIR):
    os.makedirs(IMAGE_DATA_DIR)

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

def unbundle_episode(filename, source_dir, dest_dir):
  raw_data = np.load(os.path.join(source_dir, filename))['obs']
  for i, image in enumerate(raw_data):
    filename_out = os.path.join(dest_dir, filename+"-"+str(i))
    if not os.path.exists(filename_out+".npz"):
      # print(filename_out, "doesn't exist")
      np.savez(filename_out, obs=[image])

def count_length_of_filelist(filelist, file_dir):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join(file_dir, filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return  total_length

def create_dataset(filelist, N=10, M=1000): # N is 10000 episodes, M is 1000 number of timesteps
  #TODO (Chazzz): Handle episodes which are less than M long (avoid black at end of buffer)
  data = np.zeros((M*N, 64, 64, 1), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = min(len(raw_data), M)
    if (idx+l) > (M*N):
      data[idx:M*N] = raw_data[0:M*N-idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data[0:l]
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)

  if len(data) == M*N and idx < M*N:
    data = data[:idx]
  return data

def create_dataset_lazy(filelist, N=1000*1000, M=1): #N=6000*1000
  data = []
  idx = 0
  for i in range(N):
    filename = filelist[i]
    # raw_data = np.load(os.path.join(IMAGE_DATA_DIR, filename))['obs']
    l = min(1, M*N-idx)
    raw_data_lazy = [(filename, i) for i in range(l)]
    if l == 0:
      print("premature break")
      break
    data[idx:idx+l] = raw_data_lazy
    idx += l
    if ((i+1) % 100000 == 0):
      print("loading file", i+1)

  if len(data) == M*N and idx < M*N:
    data = data[:idx]
  print(len(data))
  return data

def load_single_image(single_image):
  filename, index = single_image
  raw_data = np.load(os.path.join(IMAGE_DATA_DIR, filename))['obs']
  return raw_data[index]

def load_single_image_2(data_and_image):
  data, (filename, index) = data_and_image
  raw_data = np.load(os.path.join(IMAGE_DATA_DIR, filename))['obs']
  data = raw_data[index]

import concurrent.futures
def load_batch_lazy_parallel(imagelists):
  with concurrent.futures.ProcessPoolExecutor() as executor:
    batches = executor.map(load_batch_lazy, imagelists)
  return batches

def load_batch_lazy(imagelist):
  data = np.zeros((len(imagelist), 64, 64, 1), dtype=np.uint8)
  # with concurrent.futures.ProcessPoolExecutor() as executor:
    # executor.map(load_single_image_2, zip(data, imagelist))
    # for i, raw_data in enumerate(executor.map(load_single_image, imagelist)):
      # data[i] = raw_data
  # print(np.array_equal(data, np.zeros((len(imagelist), 64, 64, 1), dtype=np.uint8)))
  for i, (filename, index) in enumerate(imagelist):
    raw_data = np.load(os.path.join(IMAGE_DATA_DIR, filename))['obs']
    data[i] = raw_data[index]
  return data

def render_dataset(dataset):
  from gym.envs.classic_control import rendering
  import time
  viewer = rendering.SimpleImageViewer()
  for image in dataset[:1000]: #Don't actually render entire dataset
    print(np.stack((np.squeeze(image, axis=2),)*3, axis=-1).shape)
    viewer.imshow(np.stack((np.squeeze(image, axis=2),)*3, axis=-1))
    time.sleep(1/60)

# load dataset from record/*. only use first 10K, sorted by filename.
lazy = True
if lazy:
  filelist = os.listdir(IMAGE_DATA_DIR)
  if len(filelist) < 6000*1000:
    print("adding more images to image data directory")
    filelist = os.listdir(DATA_DIR)
    filelist.sort()
    for file in filelist:
      unbundle_episode(file, DATA_DIR, IMAGE_DATA_DIR)
      if len(os.listdir(IMAGE_DATA_DIR)) > 6000*1000:
        raise Exception("aaaaaa")
        break
    filelist = os.listdir(IMAGE_DATA_DIR)
  print("total images:", len(filelist))
  # print("check total number of images:", count_length_of_filelist(filelist, IMAGE_DATA_DIR))
  dataset = create_dataset_lazy(filelist)
else:
  filelist = os.listdir(DATA_DIR)
  filelist.sort()
  filelist = filelist[0:10000]
  print("check total number of images:", count_length_of_filelist(filelist, DATA_DIR))
  dataset = create_dataset(filelist)

# print num_batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length/batch_size))
batch_group_size = 10000
num_batches = num_batches-num_batches%batch_group_size
print("num_batches", num_batches)

import time
t0 = time.time()
for epoch in [0]:#range(NUM_EPOCH):
  np.random.shuffle(dataset)


  # if lazy:
    # ranges = [dataset[idx*batch_size:(idx+1)*batch_size] for idx in range(num_batches)]
    # batches = load_batch_lazy_parallel(ranges)
  # for idx in range(num_batches):
    # if lazy:
      # batch = load_batch_lazy(dataset[idx*batch_size:(idx+1)*batch_size])
print(time.time()-t0)


# print(d2)
# render_dataset(dataset)
raise Exception("Don't actually want to train")

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):
  np.random.shuffle(dataset)
  if lazy:
    for batch_group in range(0, num_batches, batch_group_size):
      ranges = [dataset[idx*batch_size:(idx+1)*batch_size] for idx in range(batch_group, batch_group+batch_group_size)]
      batches = load_batch_lazy_parallel(ranges)
      for batch in batches:
        obs = batch.astype(np.float)/255.0

        feed = {vae.x: obs,}

        (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
          vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
        ], feed)
      
        if ((train_step+1) % 500 == 0):
          print("step", (train_step+1), train_loss, r_loss, kl_loss)
        if ((train_step+1) % 5000 == 0):
          vae.save_json("tf_vae/vae.json")
  else:
    for idx in range(num_batches):
      batch = dataset[idx*batch_size:(idx+1)*batch_size]

      obs = batch.astype(np.float)/255.0

      feed = {vae.x: obs,}

      (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
        vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
      ], feed)
    
      if ((train_step+1) % 500 == 0):
        print("step", (train_step+1), train_loss, r_loss, kl_loss)
      if ((train_step+1) % 5000 == 0):
        vae.save_json("tf_vae/vae.json")

# finished, final model:
vae.save_json("tf_vae/vae.json")
