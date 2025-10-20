import cv2
import numpy as np

# 1. 读取图片
img1 = cv2.imread('FZY09912.JPG', 0) # 白天图片
img2 = cv2.imread('FZY09913.JPG', 0) # 晚上图片

# 2. 初始化ORB检测器
orb = cv2.ORB_create()

# 3. 寻找关键点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. 创建匹配器并进行匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 5. 绘制匹配结果
result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
cv2.imshow('Matches', result_img)
cv2.waitKey(0)