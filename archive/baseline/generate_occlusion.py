import os
import sys
import cv2
import random
import skimage
import numpy as np
from PIL import Image, ImageDraw


class generateData(object):
	def __init__(self, occ_image_paths, save_folder):
		self.occ_image_paths = occ_image_paths
		self.save_folder = save_folder
	
	def generate_dirty_plate(self, plate_image_path):
		#img = np.array(Image.new("RGB", (116,45),(255,255,255)))
		img = cv2.imread(plate_image_path)
		if img is None:
			return
		img = cv2.resize(img, (116, 45))
		dirty_num = random.choice(range(1,4))
		dirty_type = random.choice(list(self.occ_image_paths.keys()))
		image_paths = random.sample(self.occ_image_paths[dirty_type], dirty_num)
		patch_height = random.choice(range(20,25))
		for i in range(len(image_paths)):
			patch_orig = cv2.imread(image_paths[i])
			if patch_orig is None:
				continue
			h,w,_ = patch_orig.shape
			patch_width = int((w/h) * patch_height)
			patch_resized = cv2.resize(patch_orig, (patch_width, patch_height))
			left = random.choice(range(116 - patch_width))
			top = random.choice(range(45 - patch_height))
			right = left + patch_width
			bottom = top + patch_height
			img[top:bottom, left:right, :] = patch_resized
		dst_path = os.path.join(self.save_folder, os.path.basename(plate_image_path))
		cv2.imwrite(dst_path, img)
		print(dst_path)

if __name__ == '__main__':
	orig_folder = '/home/zhangruilong/happaizhedang/plate_img/'
	occ_image_paths = {'disk': ['/home/zhangruilong/happaizhedang/patch_images/disk1.jpg', '/home/zhangruilong/happaizhedang/patch_images/disk2.jpg', '/home/zhangruilong/happaizhedang/patch_images/disk3.jpg', '/home/zhangruilong/happaizhedang/patch_images/disk4.jpg'],
			'napkin': ['/home/zhangruilong/happaizhedang/patch_images/napkin1.jpg', '/home/zhangruilong/happaizhedang/patch_images/napkin2.jpg', '/home/zhangruilong/happaizhedang/patch_images/napkin3.jpg', '/home/zhangruilong/happaizhedang/patch_images/napkin4.jpg', '/home/zhangruilong/happaizhedang/patch_images/napkin5.jpg', '/root/plate_occlusion_recognition/patch_images/napkin6.jpg'],
			'tissue': ['/home/zhangruilong/happaizhedang/patch_images/tissue1.jpg', '/home/zhangruilong/happaizhedang/patch_images/tissue2.jpg', '/home/zhangruilong/happaizhedang/patch_images/tissue3.jpg']}
	save_folder = '/home/zhangruilong/happaizhedang/occlusion/'
	GD = generateData(occ_image_paths, save_folder)
	#plate_image_path = '/root/plate_occlusion_recognition/dataset_resize/total/1_0.jpg'
	#GD. generate_dirty_plate(plate_image_path)
	for f in os.listdir(orig_folder):
		plate_image_path = os.path.join(orig_folder, f)
		GD. generate_dirty_plate(plate_image_path)
		
	






