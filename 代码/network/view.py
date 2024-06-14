import numpy as np
imgs=np.load('/data/micca2018/view/'+'imgs1.npy')
output=np.load('/data/micca2018/view/'+'output1.npy')
with open('/data/micca2018/view/view.txt', 'a') as f:
    print(output, file=f)