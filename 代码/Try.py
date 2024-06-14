import numpy as np

'''
batch_size = 8
I1 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
I2 = np.array([[[10,20],[30,40]],[[50,60],[70,80]]])
I3 = np.array([[[100,200],[300,400]],[[500,600],[700,800]]])
I4 = np.array([[[1000,2000],[3000,4000]],[[5000,6000],[7000,8000]]])
I5 = np.array([[[10000,20000],[30000,40000]],[[50000,60000],[70000,80000]]])

I = [I1,I2,I3,I4,I5]

imgs = np.array([])
num = 0
for i in I:
    num += 1
    img = i
    img = img[np.newaxis,:]
    if num == 1:
        imgs = img
    else:
        imgs = np.append(imgs, values=img, axis=0)



train = {'img':imgs[:4]}

imgs = np.concatenate(train['img'], axis=0)  # 数组拼接，所有img的所有通道都叠在一起
indices = np.arange(len(imgs))  # 索引
print(indices)
patch_size = imgs.shape[-2:]    # 每张图的尺寸

data = np.zeros((batch_size, 1, *patch_size), dtype=np.float32)
print(data)
for i, idx in enumerate(indices):      # i索引   idx值
    data[i][0] = imgs[idx]
print("data.shape = ",data.shape)
print({'data': data,'idx': indices}  ) # 字典'''