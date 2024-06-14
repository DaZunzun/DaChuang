import os
import numpy as np
import nrrd


output_size = [256,256,88]
data_path = "D:\\micca2018Data\\Training Set\\"
save_path = "D:\\micca2018Data\\Save"
num=0

for i in os.listdir(data_path):
    num += 1
    info = data_path + i
    image,options = nrrd.read(info+"\\lgemri.nrrd")    #data：保存图片的多维矩阵;   #nrrd_options：保存图片的相关信息
    label,options_ = nrrd.read(info+"\\laendo.nrrd")
    label = (label == 255).astype(np.uint8)
    w, h, d = label.shape
    # 返回label中所有非零区域（分割对象）的索引
    tempL = np.nonzero(label)
    # 分别获取非零区域在x,y,z三轴的最小值和最大值，确保裁剪图像包含分割对象
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])

    # 计算目标尺寸比分割对象多余的尺寸
    px = max(output_size[0] - (maxx - minx), 0) // 2
    py = max(output_size[1] - (maxy - miny), 0) // 2

    # 在三个方向上随机扩增
    minx = max(minx - px, 0)
    maxx = min(maxx + px, w)
    miny = max(miny - py, 0)
    maxy = min(maxy + py, h)

    if maxx-minx != output_size[0]:
        if minx == 0:
            maxx = output_size[0]
        elif maxx == w:
            minx = w-output_size[0]
        else:
            maxx = minx+output_size[0]

    if maxy-miny != output_size[1]:
        if miny == 0:
            maxy = output_size[1]
        elif maxy == h:
            miny = h-output_size[1]
        else:
            maxy = miny + output_size[1]

    # 图像归一化，转为32位浮点数（numpy默认是64位）
    image = (image - np.mean(image)) / np.std(image)
    image = image.astype(np.float32)
    # 裁剪
    image = image[minx:maxx, miny:maxy, :]
    label = label[minx:maxx, miny:maxy, :]
    print(label.shape)

    image_array = np.array(image)
    label_array = np.array(label)
    np.save("D:\\micca2018Data\\Resize\\img\\" + str(num) + ".npy", image_array)
    np.save("D:\\micca2018Data\\Resize\\lab\\" + str(num) + ".npy", label_array)
