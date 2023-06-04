#Initial information extraction layer
import os
import numpy as np
import shutil
from glob import glob
from sklearn.cluster import KMeans
import cv2
from imutils import build_montages
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(resnet50.conv1,
                                    resnet50.bn1,
                                    resnet50.relu,
                                    resnet50.maxpool,
                                    resnet50.layer1,
                                    resnet50.layer2,
                                    resnet50.layer3,
                                    resnet50.layer4)

    def forward(self, x):
        x = self.resnet(x)
        return x

net = Net().eval()
image_path = []
all_images = []
images = os.listdir('./data_test_c')

for image_name in images:
    image_path.append('./data_test_c/' + image_name)
for path in image_path:
    image = Image.open(path).convert('RGB')
    image = transforms.Resize([224,224])(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = net(image)
    image = image.reshape(-1, )
    all_images.append(image.detach().numpy())

clt = KMeans(n_clusters=3)
clt.fit(all_images)
labelIDs = np.unique(clt.labels_)
j = 0
iddz = []
for labelID in labelIDs:
    iddz.append([])
    id_dz = np.where(clt.labels_ == labelID)[0]
    id_zs = np.random.choice(id_dz, size=min(25, len(id_dz)),
        replace=False)
    show_box = []
    for i in id_zs:
        image = cv2.imread(image_path[i])
        image = cv2.resize(image, (96, 96))
        show_box.append(image)
    montage = build_montages(show_box, (96, 96), (5, 5))[0]
    title = "Type {}".format(labelID)
    cv2.imshow(title, montage)
    cv2.waitKey(0)
    for i in id_dz:
        iddz[j].append(image_path[i])
    j+=1
path = './data/'
for i in range(j):
    isExists = os.path.exists(path+str(i))
    if not isExists:
        os.makedirs(path+str(i))	
        print("%s 目录创建成功"%i)
    else:
        print("%s 目录已经存在"%i)	
        continue

def mycopyfile(srcfile,dstpath):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, dstpath + fname)       
for i in range(j):
    dst_dir = path+str(i)+'/'
    for p in iddz[i]: 
        mycopyfile(p, dst_dir)                   
