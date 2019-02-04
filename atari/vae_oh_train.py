'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
import scipy.misc
from utils import pad_num
import time
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae.vae_onehot import ConvVAEOH, reset_graph

# Hyperparameters for ConvVAE
z_size=16
batch_size=300
learning_rate=0.0001
kl_tolerance=0.5 #was 0.5

# Parameters for training
NUM_EPOCH = 100000

# poorly coded global
side_length = 8
batch_size = side_length**2
num_batches = 1


def make_onehot_dataset(width, height):
  dataset = np.zeros((width*height, height, width, 1), dtype=np.float)
  index = 0
  for i in range(height):
    for j in range(width):
      dataset[index][i][j][0] = 01.0
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
t0 = time.time()
first_100 = -1
for epoch in range(NUM_EPOCH):
  np.random.shuffle(dataset)

  for idx in range(num_batches):
    batch = dataset[idx*batch_size:(idx+1)*batch_size]

    obs = batch.astype(np.float)#/255.0

    feed = {vae.x: obs,}

    # (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
      # vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
    # ], feed)
    (train_loss, r_loss, rls, x, y, train_step, _) = vae.sess.run([
      vae.loss, vae.r_loss, vae.r_loss_mean, vae.x, vae.y, vae.global_step, vae.train_op
    ], feed)
  
    if ((train_step+1) % 500 == 0):
      print("step", (train_step+1), train_loss, r_loss)#, kl_loss)
    if ((train_step+1) % 5000 == 0):
      squares_hit = 0
      for i in range(len(y)):
        for j in range(len(x)):
          if y[i][j//side_length][j%side_length] > 0.5 and x[i][j//side_length][j%side_length] > 0.5:
            squares_hit += 1
      # print("time:", time.time()-t0)
      print(time.time()-t0, "total hot:", squares_hit, "/", side_length**2)
      # print(x[0], y[0])
      vae.save_json("tf_vae/vae_oh.json")
      if squares_hit == side_length**2:
        first_100 = train_step+1
        break
  if first_100 != -1:
    break

# finished, final model:
vae.save_json("tf_vae/vae_oh.json")

# output results visually:
output_dir = "vae_oh_test"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

ordered_dataset = make_onehot_dataset(side_length, side_length)
batch_z = vae.encode(ordered_dataset)
# print("z for all 64 outputs")
# print(batch_z)
reconstruct = vae.decode(batch_z)
print((255.*reconstruct[0]).reshape(side_length, side_length))
squares_hit = 0
missing = []
for i in range(len(ordered_dataset)):
  if reconstruct[i][i//side_length][i%side_length] > 0.5:
    squares_hit += 1
  else:
    missing.append(i)
  scipy.misc.toimage(ordered_dataset[i].reshape(side_length, side_length), cmin=0.0, cmax=1.0).save(output_dir+'/%s.png' % pad_num(i))
  scipy.misc.toimage(reconstruct[i].reshape(side_length, side_length), cmin=0.0, cmax=1.0).save(output_dir+'/%s_vae.png' % pad_num(i))
  # imsave(output_dir+'/%s.png' % pad_num(i), 255.*ordered_dataset[i].reshape(side_length, side_length))
  # imsave(output_dir+'/%s_vae.png' % pad_num(i), reconstruct[0].reshape(side_length, side_length))
print("total hot:", squares_hit, "/", side_length**2)
print("missing:", missing)