import cv2
import numpy as np


# open操作：假设二值图像存储在image变量中
# 使用一个5x5大小的卷积核对二值图像进行Open操作
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# close操作：用于填补二值图像中的空洞
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 随机膨胀：用于增加二值图像中的白色区域的大小
# 生成一个随机的5x5大小的二值卷积核对二值图像进行随机膨胀操作
# （这里也可以不用随机的，因为网上说随机的卷积核可能会出错，直接把第二行注释掉就行）
kernel = np.ones((5,5), np.uint8)
dilation_kernel = np.random.choice([0, 1], size=(5, 5))
dilation = cv2.dilate(image, dilation_kernel)

# 随机侵蚀：用于减少二值图像中的白色区域的大小
kernel = np.ones((5,5), np.uint8)
erosion_kernel = np.random.choice([0, 1], size=(5, 5))
erosion = cv2.erode(image, erosion_kernel)
