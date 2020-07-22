import numpy as np
import tensorflow as tf
import lpips_tf

import sys
import warnings
import subprocess

from prettytable import PrettyTable
#from tabulate import tabulate 

if not sys.warnoptions:
    warnings.simplefilter("ignore")

#subprocess.run(["C:/Users/merle/Documents/VectorMedianFilter_Build/Debug/VectorMedianFilter.exe"])
#subprocess.run(["C:/Users/merle/Documents/VectorMedianFilter_Build/Debug/VectorMedianFilter.exe"])
#subprocess.run(["C:/Users/merle/Documents/ColorMedianFilter_Build/Debug/ColorMedianFilter.exe"])

tf.enable_eager_execution()

#Example: https://www.tensorflow.org/api_docs/python/tf/image/ssim
img_path1 = tf.io.read_file('C:/Users/merle/Pictures/img/Scene_10/original.jpg')
img1 = tf.image.decode_jpeg(img_path1, channels=3)
img_path2 = tf.io.read_file('C:/Users/merle/Pictures/img/Scene_10/Upscaled/vmf_upscaled.png')
img2 = tf.image.decode_png(img_path2, channels=3)
img_path3 = tf.io.read_file('C:/Users/merle/Pictures/img/Scene_10/Upscaled/bicubic_upscaled.png')
img3 = tf.image.decode_png(img_path3, channels=3)
img_path4 = tf.io.read_file('C:/Users/merle/Pictures/img/Scene_10/Upscaled/content-adaptive_upscaled.png')
img4 = tf.image.decode_png(img_path4, channels=3)
img_path5 = tf.io.read_file('C:/Users/merle/Pictures/img/Scene_10/Upscaled/perceptual_upscaled.png')
img5 = tf.image.decode_png(img_path5, channels=3)
img_path6 = tf.io.read_file('C:/Users/merle/Pictures/img/Scene_10/Upscaled/avmf_upscaled.png')
img6 = tf.image.decode_png(img_path6, channels=3)
img_path7 = tf.io.read_file('C:/Users/merle/Pictures/img/Scene_10/Upscaled/cmf_upscaled.png')
img7 = tf.image.decode_png(img_path7, channels=3)

names = ['Vector Median Filter','Bicubic','Content-Adaptive','Perceptual','Anti Vector Median Filter','Color Median Filter']

w = [img2.shape[1]]
h = [img2.shape[0]]

ssim_result1 = tf.image.ssim(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result2 = tf.image.ssim(img1, img3, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result3 = tf.image.ssim(img1, img4, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result4 = tf.image.ssim(img1, img5, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result5 = tf.image.ssim(img1, img6, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result6 = tf.image.ssim(img1, img7, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim = [ssim_result1.numpy(),ssim_result2.numpy(),ssim_result3.numpy(),ssim_result4.numpy(),ssim_result5.numpy(),ssim_result6.numpy()]



image0 = tf.image.per_image_standardization(img1)
image1 = tf.image.per_image_standardization(img2)
image2 = tf.image.per_image_standardization(img3)
image3 = tf.image.per_image_standardization(img4)
image4 = tf.image.per_image_standardization(img5)
image5 = tf.image.per_image_standardization(img6)
image6 = tf.image.per_image_standardization(img7)
#image6 = np.random.random(img7.shape)

tf.disable_eager_execution()

image0_ph = tf.compat.v1.placeholder(tf.float32)
image1_ph = tf.compat.v1.placeholder(tf.float32)
image2_ph = tf.compat.v1.placeholder(tf.float32)
image3_ph = tf.compat.v1.placeholder(tf.float32)
image4_ph = tf.compat.v1.placeholder(tf.float32)
image5_ph = tf.compat.v1.placeholder(tf.float32)
image6_ph = tf.compat.v1.placeholder(tf.float32)

distance_t1 = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')
distance_t2 = lpips_tf.lpips(image0_ph, image2_ph, model='net-lin', net='alex')
distance_t3 = lpips_tf.lpips(image0_ph, image3_ph, model='net-lin', net='alex')
distance_t4 = lpips_tf.lpips(image0_ph, image4_ph, model='net-lin', net='alex')
distance_t5 = lpips_tf.lpips(image0_ph, image5_ph, model='net-lin', net='alex')
distance_t6 = lpips_tf.lpips(image0_ph, image6_ph, model='net-lin', net='alex')

with tf.compat.v1.Session() as session:
    distance1 = session.run(distance_t1, feed_dict={image0_ph: image0.numpy(), image1_ph: image1.numpy()})
    distance2 = session.run(distance_t2, feed_dict={image0_ph: image0.numpy(), image2_ph: image2.numpy()})
    distance3 = session.run(distance_t3, feed_dict={image0_ph: image0.numpy(), image3_ph: image3.numpy()})
    distance4 = session.run(distance_t4, feed_dict={image0_ph: image0.numpy(), image4_ph: image4.numpy()})
    distance5 = session.run(distance_t5, feed_dict={image0_ph: image0.numpy(), image5_ph: image5.numpy()})
    distance6 = session.run(distance_t6, feed_dict={image0_ph: image0.numpy(), image6_ph: image6.numpy()})

lp = [distance1, distance2, distance3, distance4, distance5, distance6]

#Example PrettyTables: https://www.youtube.com/watch?v=gryi-dcF_mY
#Dokumentation PrettyTables: https://ronisbr.github.io/PrettyTables.jl/v0.7/ 

table = PrettyTable(['Name', 'SSIM', 'LPIPS'])
for x in range(0,6):
    table.add_row([names[x],ssim[x],lp[x]])
print(table)

#print(tabulate([["Name","Age"],["Alice",24],["Bob",19]],headers="firstrow")

input("Press enter to exit")

