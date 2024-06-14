import numpy as np
import matplotlib.pyplot as plt

# 从npy文件中读取数据
data = np.load('/data/micca2018/c1/labs.npy')
data=data[0,:,:,:]
data=np.squeeze(data)
for i in range(192):
    print(data[30,i,:])
# 将数据转换为三维数组
# data = np.transpose(data)
#
# # 显示图像
# plt.imshow(data)
#
# # 显示图像
# plt.show()
