import os
import torch
import numpy as np
import h5py
import nrrd
import glob
import itk
c = np.array([])
a = np.array([[10,20],[30,40]])
b = np.array([[60,70],[80,90]])
c = a[np.newaxis,:]
print(c)
c = np.append(c, b[np.newaxis,:], axis=0)
print(c)


'''
image_list = []
root_path='D:\micca2018Data\Training Set'
for subfolder in os.listdir(root_path):
    subfolder_path=os.path.join(root_path,subfolder)
    if os.path.isdir(subfolder_path):
        image_files=glob.glob(os.path.join(subfolder_path,'*.nrrd'))
    for i in range(2):
        image_path = image_files[i]
        image = itk.imread(image_path)
        image_list.append(image)

resampled_list = []
for image in image_list:
resampler = itk.ResampleImageFilter.New(image)
new_spacing = [0.6, 0.6, 1.25]
resampler.SetOutputSpacing(new_spacing)
resampler.SetSize(image.GetLargestPossibleRegion().GetSize())
resampler.SetOutputOrigin(image.GetOrigin())
resampler.SetOutputDirection(image.GetDirection())
resampler.Update()
resampled = resampler.GetOutput()
resampled_list.append(resampled)
'''


'''
img = np.array([[[10,100],[20,200],[30,300]],[[1,10],[2,20],[30,30]]])
a = np.array([[100,1000],[200,2000],[300,3000]])
print(img)
img = np.append(img, values = [a],axis=0)
print(img)

data = {'imgs':img}
I = np.concatenate(data['imgs'],axis=0)
print(I)
'''
'''

data_path = "D:\\micca2018Data\\Training Set\\"
save_path = "D:\\micca2018Data\\Save"
num=0
for i in os.listdir(data_path):
    num+=1
    info = data_path + i
    data,options = nrrd.read(info+"\\lgemri.nrrd")    #data：保存图片的多维矩阵;   #nrrd_options：保存图片的相关信息
    labe,options_ = nrrd.read(info+"\\laendo.nrrd")
    image_array = np.array(data)
    labe_array = np.array(labe)
    np.save("D:\\micca2018Data\\Save\\img\\"+str(num)+".npy", image_array)
    np.save("D:\\micca2018Data\\Save\\lab\\"+str(num)+".npy", labe_array)
    if num>=10:
        break
'''
'''
img_path = 'D:\\micca2018Data\\Save\\img\\'
lab_path = 'D:\\micca2018Data\\Save\\lab\\'
Images1 = []
Images2 = []
Images3 = []
Images4 = []
Labels1 = []
Labels2 = []
Labels3 = []
Labels4 = []
num = 0
for i in os.listdir(img_path):
    num += 1
    data = np.load(img_path + i)
    if num <= 25:
        Images1.append(data)
    elif num <= 50:
        Images2.append(data)
    elif num <= 75:
        Images3.append(data)
    else:
        Images4.append(data)
num = 0
for i in os.listdir(lab_path):
    num += 1
    data = np.load(lab_path + i)
    if num <= 25:
        Labels1.append(data)
    elif num <= 50:
        Labels2.append(data)
    elif num <= 75:
        Labels3.append(data)
    else:
        Labels4.append(data)

Images1_array = np.array(Images1)
Images2_array = np.array(Images2)
Images3_array = np.array(Images3)
Images4_array = np.array(Images4)
Labels1_array = np.array(Labels1)
Labels2_array = np.array(Labels2)
Labels3_array = np.array(Labels3)
Labels4_array = np.array(Labels4)


np.save("D:\\micca2018Data\\c1\\imgs.npy", Images1)
np.save("D:\\micca2018Data\\c1\\labs.npy", Labels1)
np.save("D:\\micca2018Data\\c2\\imgs.npy", Images2)
np.save("D:\\micca2018Data\\c2\\labs.npy", Labels2)
np.save("D:\\micca2018Data\\cl_di\\imgs.npy", Images3)
np.save("D:\\micca2018Data\\cl_di\\labs.npy", Labels3)
np.save("D:\\micca2018Data\\op_er\\imgs.npy", Images4)
np.save("D:\\micca2018Data\\op_er\\labs.npy", Labels4)'''