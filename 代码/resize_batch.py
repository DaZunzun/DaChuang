import os
import numpy as np
import nrrd
import cv2
import nibabel as nib


output_size = [192,192,64]   # resize的尺寸

data_path = "D:\\micca2018Data\\Testing Set\\"     # 储存原图的文件路径

# 选择data_path里Range_down~Range_up范围的样本执行resize等操作
Range_down = 1
Range_up   = 50

# 选择多少个样本组成imgs.npy和labs.npy
center_size = 30

save_path_img = "D:\\DataSet_MICCA\\Resize_test\\64_192_192\\img\\"     # 储存resize之后的img.npy
save_path_lab = "D:\\DataSet_MICCA\\Resize_test\\64_192_192\\lab\\"     # 储存resize之后的lab.npy
save_path_nrrd = "D:\\DataSet_MICCA\\Resize_test\\64_192_192\\NRRD\\"   # 储存resize之后的nrrd格式的img.nrrd和lab.nrrd

save_path_imgs = "D:\\DataSet_MICCA\\Batch_test\\30_64_192_192\\imgs.npy"   # 储存拼接完成后的imgs.npy  (即某一个中心的数据)
save_path_labs = "D:\\DataSet_MICCA\\Batch_test\\30_64_192_192\\labs.npy"   # 储存拼接完成后的labs.npy


order = 0   # 对结果编号

# resize以及随机膨胀，将通道放前面等操作
num = 0     # 用来找到Range_down~Range_up范围的样本
for i in os.listdir(data_path):
    num += 1

    if num < Range_down:
        continue

    if num > Range_up:
        break

    info = data_path + i     # 某个样本的img.nrrd和lab.nrrd所在的文件夹

    # 读取nrrd格式的文件
    image, o1 = nrrd.read(info + "\\lgemri.nrrd")     # 原图
    label, o2 = nrrd.read(info + "\\laendo.nrrd")     # 标签
    label = (label == 255).astype(np.uint8)

    # 读取nib格式的文件
    # image = nib.load(info+"\\img.nii")
    # label = nib.load(info+"\\lab.nii")
    # image = image.get_fdata()
    # label = label.get_fdata()
    # label = (label == 255).astype(np.uint8)

    # label的尺寸
    w, h, d = label.shape
    # 返回label中所有非零区域（分割对象）的索引
    tempL = np.nonzero(label)

    # 分别获取非零区域在x,y,z三轴的最小值和最大值，确保裁剪图像包含分割对象
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])
    minz, maxz = np.min(tempL[2]), np.max(tempL[2])

    # print([maxx-minx, maxy-miny, maxz-minz])

    # 放弃过大的样本
    if (maxx-minx > output_size[0]) or (maxy-miny > output_size[1]) or (maxz-minz > output_size[2]):
        continue

    # 计算目标尺寸比分割对象多余的尺寸
    px = max(output_size[0] - (maxx - minx), 0) // 2
    py = max(output_size[1] - (maxy - miny), 0) // 2
    pz = max(output_size[2] - (maxz - minz), 0) // 2

    # 在三个方向上扩增到规定尺寸
    minx = max(minx - px, 0)
    maxx = min(maxx + px, w)
    miny = max(miny - py, 0)
    maxy = min(maxy + py, h)
    minz = max(minz - pz, 0)
    maxz = min(maxz + pz, d)

    # 确保尺寸是要求的尺寸
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

    if maxz-minz != output_size[2]:
        if minz == 0:
            maxz = output_size[2]
        elif maxz == d:
            minz = d-output_size[2]
        else:
            maxz = minz + output_size[2]

    # 图像归一化，转为32位浮点数（numpy默认是64位）
    image = (image - np.mean(image)) / np.std(image)
    image = image.astype(np.float32)

    # 裁剪
    image = image[minx:maxx, miny:maxy, minz:maxz]
    label = label[minx:maxx, miny:maxy, minz:maxz]
    print(label.shape)
    print(image.shape)

    image_array = np.array(image)
    label_array = np.array(label)


    # 随机膨胀or随机侵蚀
    # for k in range(output_size[2]):
    #
    #     lab = label_array[:, :, k]
    #
    #     # 随机侵蚀
    #     erosion_kernel = np.random.randint(0, 8, (8, 8), dtype=np.uint8)
    #     lab = cv2.erode(lab, erosion_kernel)
    #
    #     # 打开
    #     kernel = np.ones((5, 5), np.uint8)
    #     lab = cv2.morphologyEx(lab, cv2.MORPH_OPEN, kernel)
    #
    #
    #     # 随机膨胀
    #     dilation_kernel = np.random.randint(0, 8, (8, 8), dtype=np.uint8)
    #     lab = cv2.dilate(lab, dilation_kernel)
    #
    #     # 关闭
    #     kernel = np.ones((5, 5), np.uint8)
    #     lab = cv2.morphologyEx(lab, cv2.MORPH_CLOSE, kernel)
    #
    #     label_array[:, :, k] = lab

    # 将通道放前面
    image_resize = np.zeros((output_size[2], output_size[0], output_size[1]), np.float32)
    label_resize = np.zeros((output_size[2], output_size[0], output_size[1]), np.uint8)

    for k in range(output_size[2]):
        img = image_array[:, :, k]
        lab = label_array[:, :, k]

        image_resize[k, :, :] = img
        label_resize[k, :, :] = lab

    print(image_resize.shape)
    print(label_resize.shape)

    order += 1

    # 保存修改之后的文件
    np.save(save_path_img + str(order) + ".npy", image_resize)
    np.save(save_path_lab + str(order) + ".npy", label_resize)
    nrrd.write(save_path_nrrd + "img\\" + str(order) + ".nrrd", image_resize)
    nrrd.write(save_path_nrrd + "lab\\" + str(order) + ".nrrd", label_resize)


# 在以上处理好的样本种选择center_size个样本组成一个中心（或者test）的数据（即进行拼接操作）
print("拼接！！！")

# 先初始化imgs.npy和labs.npy
imgs = np.array([])
labs = np.array([])

# 拼接得到imgs.npy
num = 0
for i in os.listdir(save_path_img):
    num += 1
    if num > center_size:
        break

    img = np.load(save_path_img + i)    # 读取到某个img.npy文件
    print(img.shape)
    img = img[np.newaxis, :]
    if num == 1:
        imgs = img
    else:
        imgs = np.append(imgs, values=img, axis=0)


# 拼接得到labs.npy文件
num = 0
for i in os.listdir(save_path_lab):
    num += 1
    if num > center_size:
        break

    lab = np.load(save_path_lab + i)   # 读取到某个lab.npy文件
    lab = lab[np.newaxis, :]
    if num == 1:
        labs = lab
    else:
        labs = np.append(labs, values=lab, axis=0)

# 输出拼接完成之后的尺寸
print(imgs.shape)
print(labs.shape)

# 储存该中心的数据imgs.npy和labs.npy
np.save(save_path_imgs, imgs)
np.save(save_path_labs, labs)
