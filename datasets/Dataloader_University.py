import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image

class Dataloader_University(Dataset):
    def __init__(self,root,transforms,names=['satellite','street','drone','google']):
        super(Dataloader_University).__init__()
        self.transforms_drone_street = transforms['train']
        self.transforms_satellite = transforms['satellite']
        self.root = root
        self.names =  names
        dict_path = {}
        for name in names:
            dict_ = {}
            for cls_name in os.listdir(os.path.join(root, name)):
                img_list = os.listdir(os.path.join(root,name,cls_name))
                img_path_list = [os.path.join(root,name,cls_name,img) for img in img_list]
                dict_[cls_name] = img_path_list
            dict_path[name] = dict_
            # dict_path[name+"/"+cls_name] = img_path_list

        cls_names = os.listdir(os.path.join(root,names[0]))
        cls_names.sort()
        map_dict={i:cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        self.index_cls_nums = 2

    def sample_from_cls(self,name,cls_num):
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path,1)[0]
        img = Image.open(img_path)
        return img


    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        img = self.sample_from_cls("satellite",cls_nums)
        img_s = self.transforms_satellite(img)

        img = self.sample_from_cls("street",cls_nums)
        img_st = self.transforms_drone_street(img)

        img = self.sample_from_cls("drone",cls_nums)
        img_d = self.transforms_drone_street(img)
        return img_s,img_st,img_d,index


    def __len__(self):
        return len(self.cls_names)



class Sampler_University(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batchsize=8,sample_num=4,triplet_loss=0):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num
        self.triplet_loss = triplet_loss

    def __iter__(self):
        list = np.arange(0,self.data_len)
        nums = np.repeat(list,self.sample_num,axis=0)
        np.random.shuffle(nums)
        # print(nums)
        return iter(nums)

    def __len__(self):
        return len(self.data_source)


def train_collate_fn(batch):
    img_s,img_st,img_d,ids = zip(*batch)
    ids = torch.tensor(ids,dtype=torch.int64)
    return [torch.stack(img_s, dim=0),ids],[torch.stack(img_st,dim=0),ids], [torch.stack(img_d,dim=0),ids]

if __name__ == '__main__':
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]


    transform_train_list ={"satellite": transforms.Compose(transform_train_list),
                            "train":transforms.Compose(transform_train_list)}
    datasets = Dataloader_University(root="E:/crossview/code/data/train",transforms=transform_train_list,names=['satellite','street','drone'])
    samper = Sampler_University(datasets,8,2)
    dataloader = DataLoader(datasets,batch_size=8,num_workers=0,sampler=samper,collate_fn=train_collate_fn)
    for data_s,data_st,data_d in dataloader:
        inputs, labels = data_s
        inputs2, labels2 = data_st
        inputs3, labels3 = data_d
        print(labels, labels2, labels3)



