'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
from scipy.misc import imsave
from utils import pad_num
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae.vae_onehot import ConvVAEOH, reset_graph

# Hyperparameters for ConvVAE
z_size=32
batch_size=300
learning_rate=0.0001
kl_tolerance=0.05 #was 0.5

# Parameters for training
NUM_EPOCH = 20000

# poorly coded global
side_length = 8
batch_size = side_length**2
num_batches = 1


def make_onehot_dataset(width, height):
  dataset = np.zeros((width*height, height, width, 1), dtype=np.float)
  index = 0
  for i in range(height):
    for j in range(width):
      dataset[index][i][j][0] = 1.0
      index += 1
  return dataset

dataset = make_onehot_dataset(side_length, side_length)

reset_graph()

vae = ConvVAEOH(z_size=z_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                kl_tolerance=kl_tolerance,
                is_training=True,
                reuse=False,
                gpu_mode=True)

load_checkpoint = False
model_path_name = "tf_vae"
if load_checkpoint:
  print("loading checkpoint")
  vae.load_json(os.path.join(model_path_name, 'vae_oh.json'))

# train loop:
print("train", "step", "loss", "recon_loss")
num_right = 0
for epoch in range(NUM_EPOCH):
  np.random.shuffle(dataset)

  for idx in range(num_batches):
    batch = dataset[idx*batch_size:(idx+1)*batch_size]

    obs = batch.astype(np.float)/255.0

    feed = {vae.x: obs,}

    # (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
      # vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
    # ], feed)
    (train_loss, r_loss, train_step, _) = vae.sess.run([
      vae.loss, vae.r_loss, vae.global_step, vae.train_op
    ], feed)
  
    if ((train_step+1) % 500 == 0):
      print("step", (train_step+1), train_loss, r_loss)#, kl_loss)
    if ((train_step+1) % 5000 == 0):
      vae.save_json("tf_vae/vae_oh.json")

# finished, final model:
vae.save_json("tf_vae/vae_oh.json")

# output results visually:
output_dir = "vae_oh_test"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

ordered_dataset = make_onehot_dataset(side_length, side_length)
batch_z = vae.encode(ordered_dataset)
reconstruct = vae.decode(batch_z)
for i in range(len(ordered_dataset)):
  imsave(output_dir+'/%s.png' % pad_num(i), 255.*ordered_dataset[i].reshape(side_length, side_length))
  imsave(output_dir+'/%s_vae.png' % pad_num(i), 255.*reconstruct[i].reshape(side_length, side_length))