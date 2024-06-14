import numpy as np
import os

img_path = "D:\\micca2018Data\\Resize\\img\\"
lab_path = "D:\\micca2018Data\\Resize\\lab\\"

imgs_1 = np.array([])
imgs_2 = np.array([])
labs_1 = np.array([])
labs_2 = np.array([])
test_img = np.array([])
test_lab = np.array([])

num = 0
for i in os.listdir(img_path):
    num += 1
    img = np.load(img_path + i)
    img = img[np.newaxis,:]
    if num == 1:
        imgs_1 = img
    elif num <= 10:
        imgs_1 = np.append(imgs_1, values=img, axis=0)
    elif num == 11:
        imgs_2 = img
    elif num <= 20:
        imgs_2 = np.append(imgs_2, values=img, axis=0)
    elif num == 21:
        test_img = img
    elif num <= 30:
        test_img = np.append(test_img, values=img, axis=0)
    else:
        break

num = 0
for i in os.listdir(lab_path):
    num += 1
    lab = np.load(lab_path + i)
    lab = lab[np.newaxis, :]
    if num == 1:
        labs_1 = lab
    elif num <= 10:
        labs_1 = np.append(labs_1, values=lab, axis=0)
    elif num == 11:
        labs_2 = lab
    elif num <= 20:
        labs_2 = np.append(labs_2, values=lab, axis=0)
    elif num == 21:
        test_lab = lab
    elif num <= 30:
        test_lab = np.append(test_lab, values=lab, axis=0)
    else:
        break

np.save("D:\\micca2018Data\\c1\\imgs.npy", imgs_1)
np.save("D:\\micca2018Data\\c1\\labs.npy", labs_1)
np.save("D:\\micca2018Data\\c2\\imgs.npy", imgs_2)
np.save("D:\\micca2018Data\\c2\\labs.npy", labs_2)
np.save("D:\\micca2018Data\\test\\imgs.npy", test_img)
np.save("D:\\micca2018Data\\test\\labs.npy", test_lab)
