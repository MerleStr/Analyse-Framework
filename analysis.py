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
subprocess.run(["C:/Users/merle/Documents/ColorMedianFilter_Build/Debug/ColorMedianFilter.exe"])
#subprocess.run(["C:/Users/merle/Documents/ColorMedianFilter_Build/Debug/ColorMedianFilter.exe"])
#subprocess.run(["C:/Users/merle/Documents/ColorMedianFilter_Build/Debug/ColorMedianFilter.exe"])
#subprocess.run(["C:/Users/merle/Documents/ColorMedianFilter_Build/Debug/ColorMedianFilter.exe"])

tf.enable_eager_execution()

scene = "Scene_20"

#Example: https://www.tensorflow.org/api_docs/python/tf/image/ssim
img_path1 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/original.jpg')
img1 = tf.image.decode_jpeg(img_path1, channels=3)
img_path2 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/vmf_upscaled.png')
img2 = tf.image.decode_png(img_path2, channels=3)
img_path3 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/avmf_upscaled.png')
img3 = tf.image.decode_png(img_path3, channels=3)
img_path4 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/cmf4_upscaled.png')
img4 = tf.image.decode_png(img_path4, channels=3)
img_path5 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/cmf4b_upscaled.png')
img5 = tf.image.decode_png(img_path5, channels=3)
img_path6 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/cmf4c_upscaled.png')
img6 = tf.image.decode_png(img_path6, channels=3)
img_path7 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/cmf9_upscaled.png')
img7 = tf.image.decode_png(img_path7, channels=3)
img_path8 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/bicubic_upscaled.png')
img8 = tf.image.decode_png(img_path8, channels=3)
img_path9 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/content-adaptive_upscaled.png')
img9 = tf.image.decode_png(img_path9, channels=3)
img_path10 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/perceptual_upscaled.png')
img10 = tf.image.decode_png(img_path10, channels=3)
img_path11 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/generalized_upscaled.png')
img11 = tf.image.decode_png(img_path11, channels=3)
img_path12 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/lanczos_upscaled.png')
img12 = tf.image.decode_png(img_path12, channels=3)
img_path13 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/subsampling_upscaled.png')
img13 = tf.image.decode_png(img_path13, channels=3)
img_path14 = tf.io.read_file('C:/Users/merle/Pictures/img/' + scene + '/Upscaled/bilateral_upscaled.png')
img14 = tf.image.decode_png(img_path14, channels=3)

names = ['Vector Median Filter','Anti Vector Median Filter','Color Median Filter with 4 Pixel, taking the second Pixel','Color Median Filter with 4 Pixel, taking the third Pixel','Color Median Filter with 4 Pixel, taking the average of second and third Pixel','Color Median Filter with 9 Pixel','Bicubic','Content-Adaptive','Perceptual','Generalized','Lanczos','Subsampling','Bilateral']

w = [img2.shape[1]]
h = [img2.shape[0]]

im1 = tf.image.convert_image_dtype(img1, tf.float32)
im2 = tf.image.convert_image_dtype(img2, tf.float32)
im3 = tf.image.convert_image_dtype(img3, tf.float32)
im4 = tf.image.convert_image_dtype(img4, tf.float32)
im5 = tf.image.convert_image_dtype(img5, tf.float32)
im6 = tf.image.convert_image_dtype(img6, tf.float32)
im7 = tf.image.convert_image_dtype(img7, tf.float32)
im8 = tf.image.convert_image_dtype(img8, tf.float32)
im9 = tf.image.convert_image_dtype(img9, tf.float32)
im10 = tf.image.convert_image_dtype(img10, tf.float32)
im11 = tf.image.convert_image_dtype(img11, tf.float32)
im12 = tf.image.convert_image_dtype(img12, tf.float32)
im13 = tf.image.convert_image_dtype(img13, tf.float32)
im14 = tf.image.convert_image_dtype(img14, tf.float32)

psnr_result1 = tf.image.psnr(im1, im2, max_val=1.0)
psnr_result2 = tf.image.psnr(im1, im3, max_val=1.0)
psnr_result3 = tf.image.psnr(im1, im4, max_val=1.0)
psnr_result4 = tf.image.psnr(im1, im5, max_val=1.0)
psnr_result5 = tf.image.psnr(im1, im6, max_val=1.0)
psnr_result6 = tf.image.psnr(im1, im7, max_val=1.0)
psnr_result7 = tf.image.psnr(im1, im8, max_val=1.0)
psnr_result8 = tf.image.psnr(im1, im9, max_val=1.0)
psnr_result9 = tf.image.psnr(im1, im10, max_val=1.0)
psnr_result10 = tf.image.psnr(im1, im11, max_val=1.0)
psnr_result11 = tf.image.psnr(im1, im12, max_val=1.0)
psnr_result12 = tf.image.psnr(im1, im12, max_val=1.0)
psnr_result13 = tf.image.psnr(im1, im12, max_val=1.0)
psnr = [psnr_result1.numpy(),psnr_result2.numpy(),psnr_result3.numpy(),psnr_result4.numpy(),psnr_result5.numpy(),psnr_result6.numpy(),psnr_result7.numpy(),psnr_result8.numpy(),psnr_result9.numpy(),psnr_result10.numpy(),psnr_result11.numpy(),psnr_result12.numpy(),psnr_result13.numpy()]

ssim_result1 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result2 = tf.image.ssim(im1, im3, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result3 = tf.image.ssim(im1, im4, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result4 = tf.image.ssim(im1, im5, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result5 = tf.image.ssim(im1, im6, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result6 = tf.image.ssim(im1, im7, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result7 = tf.image.ssim(im1, im8, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result8 = tf.image.ssim(im1, im9, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result9 = tf.image.ssim(im1, im10, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result10 = tf.image.ssim(im1, im11, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result11 = tf.image.ssim(im1, im12, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result12 = tf.image.ssim(im1, im13, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim_result13 = tf.image.ssim(im1, im14, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
ssim = [ssim_result1.numpy(),ssim_result2.numpy(),ssim_result3.numpy(),ssim_result4.numpy(),ssim_result5.numpy(),ssim_result6.numpy(),ssim_result7.numpy(),ssim_result8.numpy(),ssim_result9.numpy(),ssim_result10.numpy(),ssim_result11.numpy(),ssim_result12.numpy(),ssim_result13.numpy()]

image0 = tf.image.per_image_standardization(img1)
image1 = tf.image.per_image_standardization(img2)
image2 = tf.image.per_image_standardization(img3)
image3 = tf.image.per_image_standardization(img4)
image4 = tf.image.per_image_standardization(img5)
image5 = tf.image.per_image_standardization(img6)
image6 = tf.image.per_image_standardization(img7)
image7 = tf.image.per_image_standardization(img8)
image8 = tf.image.per_image_standardization(img9)
image9 = tf.image.per_image_standardization(img10)
image10 = tf.image.per_image_standardization(img11)
image11 = tf.image.per_image_standardization(img12)
image12 = tf.image.per_image_standardization(img13)
image13 = tf.image.per_image_standardization(img14)
#image6 = np.random.random(img7.shape)

tf.disable_eager_execution()

image0_ph = tf.compat.v1.placeholder(tf.float32)
image1_ph = tf.compat.v1.placeholder(tf.float32)
image2_ph = tf.compat.v1.placeholder(tf.float32)
image3_ph = tf.compat.v1.placeholder(tf.float32)
image4_ph = tf.compat.v1.placeholder(tf.float32)
image5_ph = tf.compat.v1.placeholder(tf.float32)
image6_ph = tf.compat.v1.placeholder(tf.float32)
image7_ph = tf.compat.v1.placeholder(tf.float32)
image8_ph = tf.compat.v1.placeholder(tf.float32)
image9_ph = tf.compat.v1.placeholder(tf.float32)
image10_ph = tf.compat.v1.placeholder(tf.float32)
image11_ph = tf.compat.v1.placeholder(tf.float32)
image12_ph = tf.compat.v1.placeholder(tf.float32)
image13_ph = tf.compat.v1.placeholder(tf.float32)

distance_t1 = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')
distance_t2 = lpips_tf.lpips(image0_ph, image2_ph, model='net-lin', net='alex')
distance_t3 = lpips_tf.lpips(image0_ph, image3_ph, model='net-lin', net='alex')
distance_t4 = lpips_tf.lpips(image0_ph, image4_ph, model='net-lin', net='alex')
distance_t5 = lpips_tf.lpips(image0_ph, image5_ph, model='net-lin', net='alex')
distance_t6 = lpips_tf.lpips(image0_ph, image6_ph, model='net-lin', net='alex')
distance_t7 = lpips_tf.lpips(image0_ph, image7_ph, model='net-lin', net='alex')
distance_t8 = lpips_tf.lpips(image0_ph, image8_ph, model='net-lin', net='alex')
distance_t9 = lpips_tf.lpips(image0_ph, image9_ph, model='net-lin', net='alex')
distance_t10 = lpips_tf.lpips(image0_ph, image10_ph, model='net-lin', net='alex')
distance_t11 = lpips_tf.lpips(image0_ph, image11_ph, model='net-lin', net='alex')
distance_t12 = lpips_tf.lpips(image0_ph, image12_ph, model='net-lin', net='alex')
distance_t13 = lpips_tf.lpips(image0_ph, image13_ph, model='net-lin', net='alex')

with tf.compat.v1.Session() as session:
    distance1 = session.run(distance_t1, feed_dict={image0_ph: image0.numpy(), image1_ph: image1.numpy()})
    distance2 = session.run(distance_t2, feed_dict={image0_ph: image0.numpy(), image2_ph: image2.numpy()})
    distance3 = session.run(distance_t3, feed_dict={image0_ph: image0.numpy(), image3_ph: image3.numpy()})
    distance4 = session.run(distance_t4, feed_dict={image0_ph: image0.numpy(), image4_ph: image4.numpy()})
    distance5 = session.run(distance_t5, feed_dict={image0_ph: image0.numpy(), image5_ph: image5.numpy()})
    distance6 = session.run(distance_t6, feed_dict={image0_ph: image0.numpy(), image6_ph: image6.numpy()})
    distance7 = session.run(distance_t7, feed_dict={image0_ph: image0.numpy(), image7_ph: image7.numpy()})
    distance8 = session.run(distance_t8, feed_dict={image0_ph: image0.numpy(), image8_ph: image8.numpy()})
    distance9 = session.run(distance_t9, feed_dict={image0_ph: image0.numpy(), image9_ph: image9.numpy()})
    distance10 = session.run(distance_t10, feed_dict={image0_ph: image0.numpy(), image10_ph: image10.numpy()})
    distance11 = session.run(distance_t11, feed_dict={image0_ph: image0.numpy(), image11_ph: image11.numpy()})
    distance12 = session.run(distance_t12, feed_dict={image0_ph: image0.numpy(), image12_ph: image12.numpy()})
    distance13 = session.run(distance_t13, feed_dict={image0_ph: image0.numpy(), image13_ph: image13.numpy()})


lp = [distance1, distance2, distance3, distance4, distance5, distance6, distance7, distance8, distance9, distance10, distance11, distance12, distance13]

#Example PrettyTables: https://www.youtube.com/watch?v=gryi-dcF_mY
#Dokumentation PrettyTables: https://ronisbr.github.io/PrettyTables.jl/v0.7/ 

table = PrettyTable(['Name', 'PSNR', 'SSIM', 'LPIPS'])
for x in range(0,13):
    table.add_row([names[x],psnr[x],ssim[x],lp[x]])
print(table)

#print(tabulate([["Name","Age"],["Alice",24],["Bob",19]],headers="firstrow")

input("Press enter to exit")

