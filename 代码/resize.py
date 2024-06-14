import os
import numpy as np
import nrrd
import cv2
import nibabel as nib


output_size = [192,192,64]
data_path = "D:\\micca2018Data\\Testing Set\\"     # 储存原图的文件路径
save_path = "D:\\DataSet_MICCA\\Resize_test\\64_192_192\\"     # 储存结果的文件夹（文件夹包含img,lab,NRRD(img,lab)）

# 选择data_path里Range_down~Range_up的样本执行resize等操作
Range_down = 1
Range_up   = 50

order = 0   # 对结果编号
num = 0
for i in os.listdir(data_path):
    num += 1

    if num < Range_down:
        continue

    if num > Range_up:
        break

    info = data_path + i

    # nrrd格式
    image, o1 = nrrd.read(info + "\\lgemri.nrrd")     # 原图
    label, o2 = nrrd.read(info + "\\laendo.nrrd")     # 标签
    label = (label == 255).astype(np.uint8)

    # nib格式
    # image = nib.load(info+"\\img.nii")    #data：保存图片的多维矩阵;
    # label = nib.load(info+"\\lab.nii")
    # image = image.get_fdata()
    # label = label.get_fdata()
    # image = np.array(image)
    # label = np.array(label)
    # print(image.shape,label.shape)

    w, h, d = label.shape
    # 返回label中所有非零区域（分割对象）的索引
    tempL = np.nonzero(label)

    # 分别获取非零区域在x,y,z三轴的最小值和最大值，确保裁剪图像包含分割对象
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])
    minz, maxz = np.min(tempL[2]), np.max(tempL[2])

    # print([maxx-minx, maxy-miny, maxz-minz])

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

    np.save(save_path + "img\\" + str(order) + ".npy", image_resize)
    np.save(save_path + "lab\\" + str(order) + ".npy", label_resize)
    nrrd.write(save_path + "NRRD\\img\\" + str(order) + ".nrrd", image_resize)
    nrrd.write(save_path + "NRRD\\lab\\" + str(order) + ".nrrd", label_resize)