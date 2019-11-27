#-*- coding:utf-8 -*-
import os
import glob
import numpy as np
import tensorflow as tf
from scipy.misc import imread,imresize
from abc import abstractmethod
from utils import unpickle

CIFAR10_DATASET = 'cifar10'
PLACES365_DATASET = 'places365'

# 自定义自己的数据集

CYBER_DATASET = 'cyber'


## np.random.binomial 返回一个满足二项分布的值

class BaseDataset():
    def __init__(self, name, path, training=True, augment=True):
        self.name = name
        self.augment = augment and training
        self.training = training
        self.path = path
        self._data = []
        self._sketchdata = [] #线稿数据

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        total = len(self)
        start = 0

        while start < total:
            item = self[start]
            start += 1
            yield item

        raise StopIteration
	#getitem的get方法
    def __getitem__(self, index):
        val = self.data[index]
        try:
            img = imread(val) if isinstance(val, str) else val
            img = imresize(img,(512,512))

            #::-1  对数组进行倒序 翻转 数据增强
            if self.augment and np.random.binomial(1, 0.5) == 1:
                img = img[:, ::-1, :]

        except:
            img = None

        return img

	#getSketch的get方法
    def getsketchItem(self, index):
        val = self.sketchdata[index]
        try:
            img = imread(val) if isinstance(val, str) else val
            img = imresize(img,(512,512))

            #::-1  对数组进行倒序 翻转 数据增强
            if self.augment and np.random.binomial(1, 0.5) == 1:
                img = img[:, ::-1]

        except:
            img = None

        return img

    def generator(self, batch_size, recusrive=False):
        start = 0
        total = len(self)

        while True:
            while start < total:
                end = np.min([start + batch_size, total])
                items = []

                for ix in range(start, end):
                    orgineItem = self[ix]
                    sketchItem = self.getsketchItem(ix)
                    if orgineItem is not None and sketchItem is not None:
                        singleItem = [orgineItem,sketchItem]
                        items.append(singleItem)
                    
                np.random.shuffle(items)

                start = end
                yield items

            if recusrive:
                start = 0

            else:
                raise StopIteration

	#获取原图数据
    @property
    def data(self):
        if len(self._data) == 0:
            self._data = self.load()
            #np.random.shuffle(self._data)

        return self._data
	#获取线稿数据
    @property
    def sketchdata(self):
        if len(self._sketchdata) == 0:
            self._sketchdata = self.loadSketch()
            #np.random.shuffle(self._sketchdata)

        return self._sketchdata

    @abstractmethod
    def load(self):
        return []

    def loadSketch(self):
        return []

# 自定义自己的数据集合
class CyberDataset(BaseDataset):
    def __init__(self, path, training=True, augment=True):
        super(CyberDataset, self).__init__(CYBER_DATASET, path, training, augment)

    def load(self):
        if self.training:
            flist = os.path.join(self.path, 'train.flist')
            if os.path.exists(flist):
                data = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            else:
                data = glob.glob(self.path + 'train/orignal/*.jpg', recursive=True)
                np.savetxt(flist, data, fmt='%s')

        else:
            flist = os.path.join(self.path, 'test.flist')
            if os.path.exists(flist):
                data = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            else:
                data = np.array(glob.glob(self.path + 'test/orignal/*.jpg'))
                np.savetxt(flist, data, fmt='%s')

        return data
    def loadSketch(self):
        if self.training:
            flistSketch = os.path.join(self.path, 'trainSketch.flist')
            if os.path.exists(flistSketch):
                dataSketch = np.genfromtxt(flistSketch, dtype=np.str, encoding='utf-8')
            else:
                dataSketch = glob.glob(self.path + 'train/sketch/*.jpg', recursive=True)
                np.savetxt(flistSketch, dataSketch, fmt='%s')

        else:
            flistSketch = os.path.join(self.path, 'testSketch.flist')
            if os.path.exists(flistSketch):
                dataSketch = np.genfromtxt(flistSketch, dtype=np.str, encoding='utf-8')
            else:
                dataSketch = glob.glob(self.path + 'test/sketch/*.jpg', recursive=True)
   
                np.savetxt(flistSketch, dataSketch, fmt='%s')

        return dataSketch


class TestDataset(BaseDataset):
    def __init__(self, path):
        super(TestDataset, self).__init__('TEST', path, training=False, augment=False)

    def __getitem__(self, index):
        path = self.data[index]
        img = imread(path)
        return path, img

    def load(self):

        if os.path.isfile(self.path):
            data = [self.path]

        elif os.path.isdir(self.path):
            data = list(glob.glob(self.path + '/*.jpg')) + list(glob.glob(self.path + '/*.png'))

        return data

