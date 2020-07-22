# encoding: utf-8

import numpy as np
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import random
import torch
from torchvision.transforms import ToPILImage
# def read_image(img_path):
#     """Keep reading image until succeed.
#     This can avoid IOError incurred by heavy IO process."""
#     got_img = False
#     if not osp.exists(img_path):
#         raise IOError("{} does not exist".format(img_path))
#     while not got_img:
#         try:
#             img = Image.open(img_path).convert('RGB')
#             got_img = True
#         except IOError:
#             print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
#             pass
#     return img
import torchvision.transforms as T
import math
from data.transforms.transforms import RandomErasing,MaskRandomErasing,RandomHorizontalFlip,RandomCrop
import random
from PIL import Image
def build_transforms(cfg, radint,i,j,is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            RandomHorizontalFlip(radint,p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            RandomCrop(cfg.INPUT.SIZE_TRAIN,i,j),
            T.ToTensor(),
            normalize_transform,
            # RandomErasing(radint,arget_area,aspect_ratio,probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform

def mask_transforms(cfg, radint,i,j,is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize((256,128)),
            RandomHorizontalFlip(radint,p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            RandomCrop(cfg.INPUT.SIZE_TRAIN,i,j),
            T.ToTensor(),
            # normalize_transform,
            # MaskRandomErasing(radint,arget_area,aspect_ratio,probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
def get_x1y1(img):
    for attempt in range(100):
        area = img.size()[1] * img.size()[2]
        target_area = random.uniform(0.02, 0.4) * area
        aspect_ratio = random.uniform(0.3, 1 / (0.3))

        # target_area = random.uniform(self.sl, self.sh) * area
        # aspect_ratio = random.uniform(self.r1, 1 / self.r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.size()[2] and h < img.size()[1]:
            x1 = random.randint(0, img.size()[1] - h)
            y1 = random.randint(0, img.size()[2] - w)
            return x1,y1,h,w

def get_image_name(img_path):
    name = img_path.split('/')[-1] #get name
    name = name[:name.rfind('.')]  #delete extention
    return name
def get_value(parsing_path):
    pred = (np.load(parsing_path))
    # print(type(pred))
    # np.where(pred > 0, pred, 1)
    pred[pred > 0] = 1
    # print(pred.shape)
    count = 0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i][j] == 1:
                count = count + 1
    # print(count)
    valu = round((count / (128 * 64)))
    return valu

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
def preprocess_salience(img):
    '''
    Resizes each image to 64 x 128 so it can be used inside architectures
    '''
    img = Image.fromarray(img)
    img = img.resize((32, 64), Image.BILINEAR)
    img = np.array(img)
    return img

def read_numpy_file(file_path):
    """Keep reading file until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_file = False
    if not osp.exists(file_path):
        raise IOError("{} does not exist".format(file_path))
    while not got_file:
        try:
            file = np.load(file_path)
            got_file = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(file_path))
            pass
    return file
def body_decode_parsing(mask):
    body_masks = torch.zeros(5, mask.size()[0], mask.size()[1])
    # foreground
    body_masks[0, mask > 0] = 1

    # head = hat, hair, sunglasses,  face
    body_masks[1, mask == 1] = 1
    body_masks[1, mask == 2] = 1
    body_masks[1, mask == 4] = 1

    body_masks[1, mask == 13] = 1

    # upper body = Upperclothes, dress, jumpsuits, leftarm, rightarm,
    body_masks[2, mask == 5] = 1
    body_masks[2, mask == 6] = 1
    body_masks[2, mask == 7] = 1
    body_masks[2, mask == 12] = 1
    body_masks[2, mask == 14] = 1
    body_masks[2, mask == 15] = 1

    # lower_body = pants, skirt, leftLeg, rightLeg
    body_masks[3, mask == 9] = 1
    body_masks[3, mask == 10] = 1
    body_masks[3, mask == 16] = 1
    body_masks[3, mask == 17] = 1

    # shoes = socks, leftshoe, rightshoe
    body_masks[4, mask == 8] = 1
    body_masks[4, mask == 18] = 1
    body_masks[4, mask == 19] = 1

    body_masks = body_masks.numpy()

    # resize to half of the original images
    masks = []
    flag=True
    for body_mask in body_masks:
         body_mask = Image.fromarray(body_mask)
         # body_mask = body_mask.resize((32, 64), Image.BILINEAR)
         body_mask = body_mask.resize((8, 16), Image.BILINEAR)
         masks.append(np.array(body_mask))
    return masks





    #     # body_mask = body_mask.resize((64, 192), Image.BILINEAR)

    #     masks.append(np.array(body_mask))
    # for i in range(len(body_masks)):
    #     body_mask = Image.fromarray(body_masks[i,:])
    #     # body_mask = body_mask.resize((64, 192), Image.BILINEAR)
    #     if i==0:
    #         body_mask = body_mask.resize((32, 64), Image.BILINEAR)
    #     else:
    #         body_mask = body_mask.resize((8, 16), Image.BILINEAR)
    #     masks.append(np.array(body_mask))
    # return np.array(masks), np.array(label)
def decode_parsing(mask):
    body_masks = torch.zeros(5, mask.size()[0], mask.size()[1])
    label = mask

    label[label == 3] = 1

    label[label == 1] = 1
    label[label == 2] = 1
    label[label == 4] = 1
    label[label == 13] = 1

    label[label == 5] = 2
    label[label == 6] = 2
    label[label == 7] = 2
    label[label == 10] = 2
    label[label == 14] = 2
    label[label == 15] = 2

    label[label == 9] = 3
    label[label == 12] = 3
    label[label == 16] = 3
    label[label == 17] = 3

    label[label == 8] =4
    label[label == 18] = 4
    label[label == 19] =4

    # foreground
    body_masks[0, mask > 0] = 1

    # head = hat, hair, sunglasses, face
    body_masks[1, mask == 1] = 1
    body_masks[1, mask == 2] = 1
    body_masks[1, mask == 4] = 1
    # body_masks[1, mask == 7] = 1
    body_masks[1, mask == 13] = 1

    # upper body = Upperclothes, dress, jumpsuits, coat,leftarm, rightarm,
    body_masks[2, mask == 3] = 1
    body_masks[2, mask == 5] = 1
    body_masks[2, mask == 6] = 1
    body_masks[2, mask == 7] = 1
    body_masks[2, mask == 10] = 1
    body_masks[2, mask == 14] = 1
    body_masks[2, mask == 15] = 1

    # lower_body = pants, skirt, leftLeg, rightLeg
    body_masks[3, mask == 9] = 1
    body_masks[3, mask == 12] = 1
    body_masks[3, mask == 16] = 1
    body_masks[3, mask == 17] = 1

    # shoes = socks, leftshoe, rightshoe
    body_masks[4, mask == 8] = 1
    body_masks[4, mask == 18] = 1
    body_masks[4, mask == 19] = 1

    body_masks = body_masks.numpy()

    # resize to half of the original images
    masks = []
    # flag=True
    # mask[mask>0]=1
    # mask=Image.fromarray(mask)
    # mask=mask.resize((32,64),Image.BILINEAR)
    # return np.array(mask,dtype=float)
    for body_mask in body_masks:
         body_mask = Image.fromarray(body_mask)
         body_mask = body_mask.resize((32, 64), Image.BILINEAR)
         masks.append(np.array(body_mask))
    # #     # body_mask = body_mask.resize((64, 192), Image.BILINEAR)
    #
    # #     masks.append(np.array(body_mask))
    # # for i in range(len(body_masks)):
    # #     body_mask = Image.fromarray(body_masks[i,:])
    # #     # body_mask = body_mask.resize((64, 192), Image.BILINEAR)
    # #     if i==0:
    # #         body_mask = body_mask.resize((32, 64), Image.BILINEAR)
    # #     else:
    # #         body_mask = body_mask.resize((8, 16), Image.BILINEAR)
    # #     masks.append(np.array(body_mask))
    # # return np.array(masks), np.array(label)
    label = np.unique(label)
    # return masks[0], np.array(label)
    return masks[0]
def label_parsing(mask):
    body_masks = torch.zeros(5, mask.size()[0], mask.size()[1])
    label = mask
    label[label>0]=1
    label[label == 3] = 2

    label[label == 1] = 2
    label[label == 2] = 2
    label[label == 4] = 2
    label[label == 13] = 2

    label[label == 5] = 3
    label[label == 6] = 3
    label[label == 7] = 3
    label[label == 10] = 3
    label[label == 14] = 3
    label[label == 15] = 3

    label[label == 9] = 4
    label[label == 12] = 4
    label[label == 16] = 4
    label[label == 17] = 4

    label[label == 8] = 5
    label[label == 18] = 5
    label[label == 19] = 5

    # foreground
    body_masks[0, mask > 0] = 1

    # head = hat, hair, sunglasses, coat, face
    body_masks[1, mask == 1] = 1
    body_masks[1, mask == 2] = 1
    body_masks[1, mask == 4] = 1
    body_masks[1, mask == 7] = 1
    body_masks[1, mask == 13] = 1

    # upper body = Upperclothes, dress, jumpsuits, leftarm, rightarm,
    body_masks[2, mask == 5] = 1
    body_masks[2, mask == 6] = 1
    body_masks[2, mask == 10] = 1
    body_masks[2, mask == 14] = 1
    body_masks[2, mask == 15] = 1

    # lower_body = pants, skirt, leftLeg, rightLeg
    body_masks[3, mask == 9] = 1
    body_masks[3, mask == 12] = 1
    body_masks[3, mask == 16] = 1
    body_masks[3, mask == 17] = 1

    # shoes = socks, leftshoe, rightshoe
    body_masks[4, mask == 8] = 1
    body_masks[4, mask == 18] = 1
    body_masks[4, mask == 19] = 1

    body_masks = body_masks.numpy()
    label=np.unique(label)

    return np.array(label)

def new_decode_parsing(mask,value):
    body_masks = torch.zeros(5, mask.size()[0], mask.size()[1])
    mask[mask==0]=value
    # import pdb
    # pdb.set_trace()
    label = mask
    mask[mask == 0] = value
    label[label == 3] = 1

    label[label == 1] = 1
    label[label == 2] = 1
    label[label == 4] = 1
    label[label == 13] = 1

    label[label == 5] = 2
    label[label == 6] = 2
    label[label == 7] = 2
    label[label == 10] = 2
    label[label == 14] = 2
    label[label == 15] = 2

    label[label == 9] = 3
    label[label == 12] = 3
    label[label == 16] = 3
    label[label == 17] = 3

    label[label == 8] = 4
    label[label == 18] = 4
    label[label == 19] = 4

    # foreground
    body_masks[0, mask > 0] = 1

    # head = hat, hair, sunglasses, coat, face
    body_masks[1, mask == 1] = 1
    body_masks[1, mask == 2] = 1
    body_masks[1, mask == 4] = 1
    # body_masks[1, mask == 7] = 1
    body_masks[1, mask == 13] = 1

    # upper body = Upperclothes, dress, jumpsuits, leftarm, rightarm,
    body_masks[2, mask == 5] = 1
    body_masks[2, mask == 6] = 1
    body_masks[2, mask == 7] = 1
    body_masks[2, mask == 10] = 1
    body_masks[2, mask == 14] = 1
    body_masks[2, mask == 15] = 1

    # lower_body = pants, skirt, leftLeg, rightLeg
    body_masks[3, mask == 9] = 1
    body_masks[3, mask == 12] = 1
    body_masks[3, mask == 16] = 1
    body_masks[3, mask == 17] = 1

    # shoes = socks, leftshoe, rightshoe
    body_masks[4, mask == 8] = 1
    body_masks[4, mask == 18] = 1
    body_masks[4, mask == 19] = 1

    body_masks = body_masks.numpy()

    # resize to half of the original images
    masks = []
    # flag=True
    for body_mask in body_masks:
         body_mask = Image.fromarray(body_mask)
         body_mask = body_mask.resize((32, 64), Image.BILINEAR)
         masks.append(np.array(body_mask))





    #     # body_mask = body_mask.resize((64, 192), Image.BILINEAR)

    #     masks.append(np.array(body_mask))
    # for i in range(len(body_masks)):
    #     body_mask = Image.fromarray(body_masks[i,:])
    #     # body_mask = body_mask.resize((64, 192), Image.BILINEAR)
    #     if i==0:
    #         body_mask = body_mask.resize((32, 64), Image.BILINEAR)
    #     else:
    #         body_mask = body_mask.resize((8, 16), Image.BILINEAR)
    #     masks.append(np.array(body_mask))
    # return np.array(masks), np.array(label)
    return masks
# class ImageDataset(Dataset):
#     """Image Person ReID Dataset"""
#
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         img_path, pid, camid = self.dataset[index]
#         img = read_image(img_path)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, pid, camid, img_path

class ValidImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, salience_base_path = 'salience/', use_salience = False, parsing_base_path = 'parsing/', use_parsing = False, transform_salience_parsing = None):
        self.dataset = dataset
        self.transform= transform
        self.use_salience = use_salience
        self.use_parsing = use_parsing
        self.salience_base_path = salience_base_path
        self.parsing_base_path = parsing_base_path
        self.transform_salience_parsing = transform_salience_parsing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        # print(img_path)
        img = read_image(img_path)
        seed = random.randint(0,2**32)
        radint=random.random()
        if self.transform is not None:
            random.seed(seed)
            img = self.transform(img)
        # img_transform=build_transforms(self.cfg,radint,is_train=True)
        # mask_transform=mask_transforms(self.cfg,radint,is_train=True)
        # img = img_transform(img)

        if self.use_salience and not self.use_parsing:
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')

            if self.transform_salience_parsing == None:
                salience_img = preprocess_salience(read_numpy_file(salience_path))
            else:
                random.seed(seed)
                salience_img = self.transform_salience_parsing(Image.fromarray(read_numpy_file(salience_path)))
                # salience_img = mask_transform(Image.fromarray(read_numpy_file(salience_path)))
                # print(salience_img.size())
                # import pdb
                # pdb.set_trace()
                salience_img = salience_img.resize((32, 64), Image.BILINEAR)
                salience_img = np.array(salience_img)
                salience_img =torch.Tensor(salience_img)

            return img, pid, camid, salience_img, img_path
        elif not self.use_salience and self.use_parsing:
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')
            parsing_path = osp.join(self.parsing_base_path, get_image_name(img_path) + '.npy')
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')
            zerovalue = get_value(parsing_path)
            # parsing_img, label_mark = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))



            # print(parsing_img.shape)
            # print(parsing_img.shape[0])
            # print(parsing_img.shape[1])
            # parsing_img=parsing_img.astype(float)
            # np.set_printoptions(precision=4)
            # salience_img = preprocess_salience(read_numpy_file(salience_path))
            # for i in range(parsing_img.shape[0]):
            #     for j in range(parsing_img.shape[1]):
            #         if parsing_img[i][j]==0:
            #             if salience_img[i][j]>0.1:
            #                 salience_img[i][j]=1
            #             parsing_img[i][j]=salience_img[i][j]
                        # print(parsing_img[i][j])
                        # import pdb
                        # pdb.set_trace()

            # parsing_img=new_decode_parsing(torch.tensor(read_numpy_file(parsing_path)),zerovalue)
            body_parsing_img = body_decode_parsing(torch.tensor(read_numpy_file(parsing_path)))

            if self.transform_salience_parsing != None:
                new_parsing_img = []
                body_new_parsing_img = []
                random.seed(seed)

                img_i = self.transform_salience_parsing(Image.fromarray(parsing_img))
                img_i = np.array(img_i)
                # print(img_i.shape)
                new_parsing_img.append(img_i)
                # for slide in parsing_img:
                #     random.seed(seed)
                #     img_i = self.transform_salience_parsing(Image.fromarray(slide))
                #     # img_i = img_i.resize((64, 192), Image.BILINEAR)
                #     # img_i = img_i.resize((32, 64), Image.BILINEAR)
                #     img_i = np.array(img_i)
                #     new_parsing_img.append(img_i)
                for slide in body_parsing_img:
                    random.seed(seed)
                    img_i = self.transform_salience_parsing(Image.fromarray(slide))
                    # img_i = img_i.resize((64, 192), Image.BILINEAR)
                    # img_i = img_i.resize((32, 64), Image.BILINEAR)
                    img_i = np.array(img_i)
                    body_new_parsing_img.append(img_i)

                # parsing_img = np.array(new_parsing_img)
                parsing_img = torch.Tensor(new_parsing_img)
                # print(parsing_img.size())
                # body_new_parsing_img = np.array(body_new_parsing_img)
                body_new_parsing_img = torch.Tensor(body_new_parsing_img)

            return img, pid, camid, parsing_img,body_new_parsing_img, img_path
            # return img, pid, camid, parsing_img,  img_path

        elif self.use_parsing and self.use_salience:
            parsing_path = osp.join(self.parsing_base_path, get_image_name(img_path) + '.npy')
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')

            if self.transform_salience_parsing == None:
                salience_img = preprocess_salience(read_numpy_file(salience_path))
                parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            else:
                random.seed(seed)
                salience_img = self.transform_salience(Image.fromarray(read_numpy_file(salience_path)))
                salience_img = salience_img.resize((64, 128), Image.BILINEAR)
                salience_img = np.array(salience_img)

                parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
                new_parsing_img = []
                for slide in parsing_img:
                    random.seed(seed)
                    img_i = self.transform_salience_parsing(Image.fromarray(slide))
                    img_i = img_i.resize((64, 128), Image.BILINEAR)
                    img_i = np.array(img_i)
                    new_parsing_img.append(img_i)
                parsing_img = np.array(new_parsing_img)

            return img, pid, camid, salience_img, parsing_img, img_path
        else:
            return img, pid, camid,camid,img_path

class LabelImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, salience_base_path = 'salience/', use_salience = False, parsing_base_path = 'parsing/', use_parsing = False, transform_salience_parsing = None):
        self.dataset = dataset
        self.transform= transform
        self.use_salience = use_salience
        self.use_parsing = use_parsing
        self.salience_base_path = salience_base_path
        self.parsing_base_path = parsing_base_path
        self.transform_salience_parsing = transform_salience_parsing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        seed = random.randint(0,2**32)
        radint=random.random()
        if self.transform is not None:
            random.seed(seed)
            img = self.transform(img)
        # img_transform=build_transforms(self.cfg,radint,is_train=True)
        # mask_transform=mask_transforms(self.cfg,radint,is_train=True)
        # img = img_transform(img)

        if self.use_salience and not self.use_parsing:
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')

            if self.transform_salience_parsing == None:
                salience_img = preprocess_salience(read_numpy_file(salience_path))
            else:
                random.seed(seed)
                salience_img = self.transform_salience_parsing(Image.fromarray(read_numpy_file(salience_path)))
                # salience_img = mask_transform(Image.fromarray(read_numpy_file(salience_path)))
                # print(salience_img.size())
                # import pdb
                # pdb.set_trace()
                salience_img = salience_img.resize((32, 64), Image.BILINEAR)
                salience_img = np.array(salience_img)
                salience_img =torch.Tensor(salience_img)

            return img, pid, camid, salience_img, img_path
        elif not self.use_salience and self.use_parsing:
            parsing_path = osp.join(self.parsing_base_path, get_image_name(img_path) + '.npy')
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')
            # print('*****')
            # print(img_path)
            # print(parsing_path)
            # zerovalue = get_value(parsing_path)
            # label_mark = label_parsing(torch.tensor(read_numpy_file(parsing_path)))
            parsing_img,label_mark = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            # print(parsing_img.shape)
            # print(parsing_img.shape[0])
            # print(parsing_img.shape[1])
            parsing_img=parsing_img.astype(float)
            np.set_printoptions(precision=4)
            salience_img = preprocess_salience(read_numpy_file(salience_path))
            for i in range(parsing_img.shape[0]):
                for j in range(parsing_img.shape[1]):
                    if parsing_img[i][j]==0:
                        if salience_img[i][j]>0.1:
                            salience_img[i][j]=1
                        parsing_img[i][j]=salience_img[i][j]
                        # print(parsing_img[i][j])
                        # import pdb
                        # pdb.set_trace()

            # parsing_img=new_decode_parsing(torch.tensor(read_numpy_file(parsing_path)),zerovalue)
            body_parsing_img = body_decode_parsing(torch.tensor(read_numpy_file(parsing_path)))

            if self.transform_salience_parsing != None:
                new_parsing_img = []
                body_new_parsing_img = []
                random.seed(seed)
                img_i = self.transform_salience_parsing(Image.fromarray(parsing_img))
                img_i = np.array(img_i)
                # print(img_i.shape)
                new_parsing_img.append(img_i)
                # for slide in parsing_img:
                #     random.seed(seed)
                #     img_i = self.transform_salience_parsing(Image.fromarray(slide))
                #     # img_i = img_i.resize((64, 192), Image.BILINEAR)
                #     # img_i = img_i.resize((32, 64), Image.BILINEAR)
                #     img_i = np.array(img_i)
                #     new_parsing_img.append(img_i)
                for slide in body_parsing_img:
                    random.seed(seed)
                    img_i = self.transform_salience_parsing(Image.fromarray(slide))
                    # img_i = img_i.resize((64, 192), Image.BILINEAR)
                    # img_i = img_i.resize((32, 64), Image.BILINEAR)
                    img_i = np.array(img_i)
                    body_new_parsing_img.append(img_i)

                # parsing_img = np.array(new_parsing_img)
                parsing_img = torch.Tensor(new_parsing_img)
                # print(parsing_img.size())
                # body_new_parsing_img = np.array(body_new_parsing_img)
                body_new_parsing_img = torch.Tensor(body_new_parsing_img)
            # import pdb
            # pdb.set_trace()
            # return img, pid, camid, parsing_img,body_new_parsing_img, img_path
            return img, pid, camid, label_mark,  img_path

        elif self.use_parsing and self.use_salience:
            parsing_path = osp.join(self.parsing_base_path, get_image_name(img_path) + '.npy')
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')

            if self.transform_salience_parsing == None:
                salience_img = preprocess_salience(read_numpy_file(salience_path))
                parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            else:
                random.seed(seed)
                salience_img = self.transform_salience(Image.fromarray(read_numpy_file(salience_path)))
                salience_img = salience_img.resize((64, 128), Image.BILINEAR)
                salience_img = np.array(salience_img)

                parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
                new_parsing_img = []
                for slide in parsing_img:
                    random.seed(seed)
                    img_i = self.transform_salience_parsing(Image.fromarray(slide))
                    img_i = img_i.resize((64, 128), Image.BILINEAR)
                    img_i = np.array(img_i)
                    new_parsing_img.append(img_i)
                parsing_img = np.array(new_parsing_img)

            return img, pid, camid, salience_img, parsing_img, img_path
        else:
            return img, pid, camid,img_path

class OccImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, salience_base_path = 'salience/', use_salience = False, parsing_base_path = 'parsing/', use_parsing = False, transform_salience_parsing = None):
        self.dataset = dataset
        self.transform= transform
        self.use_salience = use_salience
        self.use_parsing = use_parsing
        self.salience_base_path = salience_base_path
        self.parsing_base_path = parsing_base_path
        self.transform_salience_parsing = transform_salience_parsing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        seed = random.randint(0,2**32)
        radint=random.random()
        if self.transform is not None:
            random.seed(seed)
            img = self.transform(img)
        # img_transform=build_transforms(self.cfg,radint,is_train=True)
        # mask_transform=mask_transforms(self.cfg,radint,is_train=True)
        # img = img_transform(img)

        if self.use_salience and not self.use_parsing:
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')

            if self.transform_salience_parsing == None:
                salience_img = preprocess_salience(read_numpy_file(salience_path))
            else:
                random.seed(seed)
                salience_img = self.transform_salience_parsing(Image.fromarray(read_numpy_file(salience_path)))
                # salience_img = mask_transform(Image.fromarray(read_numpy_file(salience_path)))
                # print(salience_img.size())
                # import pdb
                # pdb.set_trace()
                salience_img = salience_img.resize((32, 64), Image.BILINEAR)
                salience_img = np.array(salience_img)
                salience_img =torch.Tensor(salience_img)

            return img, pid, camid, salience_img, img_path
        elif not self.use_salience and self.use_parsing:
            parsing_path = osp.join(self.parsing_base_path, get_image_name(img_path) + '.npy')
            zerovalue = get_value(parsing_path)
            parsing_img, label_mark = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            # parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            # parsing_img=new_decode_parsing(torch.tensor(read_numpy_file(parsing_path)),zerovalue)
            body_parsing_img = body_decode_parsing(torch.tensor(read_numpy_file(parsing_path)))

            if self.transform_salience_parsing != None:
                new_parsing_img = []
                body_new_parsing_img = []
                # for slide in parsing_img:
                #     random.seed(seed)
                #     img_i = self.transform_salience_parsing(Image.fromarray(slide))
                #     # img_i = img_i.resize((64, 192), Image.BILINEAR)
                #     # img_i = img_i.resize((32, 64), Image.BILINEAR)
                #     img_i = np.array(img_i)
                #     new_parsing_img.append(img_i)
                for slide in body_parsing_img:
                    random.seed(seed)
                    img_i = self.transform_salience_parsing(Image.fromarray(slide))
                    # img_i = img_i.resize((64, 192), Image.BILINEAR)
                    # img_i = img_i.resize((32, 64), Image.BILINEAR)
                    img_i = np.array(img_i)
                    body_new_parsing_img.append(img_i)

                # parsing_img = np.array(new_parsing_img)
                parsing_img = torch.Tensor(parsing_img)
                # body_new_parsing_img = np.array(body_new_parsing_img)
                body_new_parsing_img = torch.Tensor(body_new_parsing_img)

            # return img, pid, camid, parsing_img,body_new_parsing_img, img_path
            return img, pid, camid, parsing_img,  img_path

        elif self.use_parsing and self.use_salience:
            parsing_path = osp.join(self.parsing_base_path, get_image_name(img_path) + '.npy')
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')

            if self.transform_salience_parsing == None:
                salience_img = preprocess_salience(read_numpy_file(salience_path))
                parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            else:
                random.seed(seed)
                salience_img = self.transform_salience_parsing(Image.fromarray(read_numpy_file(salience_path)))
                salience_img = salience_img.resize((8, 16), Image.BILINEAR)
                salience_img = np.array(salience_img)

                parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
                parsing_img=self.transform_salience_parsing(Image.fromarray(parsing_img))
                parsing_img=parsing_img.resize((8, 16), Image.BILINEAR)
                parsing_img=np.array(parsing_img)
                new_parsing_img = []
                # for slide in parsing_img:
                #     random.seed(seed)
                #     img_i = self.transform_salience_parsing(Image.fromarray(slide))
                #     img_i = img_i.resize((32, 64), Image.BILINEAR)
                #     img_i = np.array(img_i)
                #     new_parsing_img.append(img_i)
                # parsing_img = np.array(new_parsing_img)

            return img, pid, camid, torch.Tensor(parsing_img), torch.Tensor(salience_img), img_path
        else:
            return img, pid, camid, img_path
class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, cfg=None, salience_base_path = 'salience/', use_salience = False, parsing_base_path = 'parsing/', use_parsing = False, transform_salience_parsing = None):
        self.dataset = dataset
        self.cfg = cfg
        self.use_salience = use_salience
        self.use_parsing = use_parsing
        self.salience_base_path = salience_base_path
        self.parsing_base_path = parsing_base_path
        self.transform_salience_parsing = transform_salience_parsing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        seed = random.randint(0,2**32)
        radint=random.randint(0,1)
        # print('radint',radint)
        # if self.transform is not None:
        #     random.seed(seed)
        #     img = self.transform(img)
        # area = img.size()[1] * img.size()[2]
        area = 256* 128
        i=random.randint(0,20)
        j=random.randint(0,20)
        # print('&&&*****')
        # print('i',i)
        # print('j',j)
        # print('####')
        target_area = random.uniform(0.02, 0.4) * area
        aspect_ratio = random.uniform(0.3, 1 / (0.3))


        img_transform=build_transforms(self.cfg,radint,i,j,is_train=True)
        mask_transform=mask_transforms(self.cfg,radint,i,j,is_train=True)
        img = img_transform(img)
        x1,y1,h,w=get_x1y1(img)
        # print(img.size())
        # import pdb
        # pdb.set_trace()

        REA=RandomErasing(radint, x1,y1,h,w, probability=self.cfg.INPUT.RE_PROB,
                              mean=self.cfg.INPUT.PIXEL_MEAN)
        img=REA(img)
        MREA=MaskRandomErasing(radint, x1,y1,h,w, probability=self.cfg.INPUT.RE_PROB,
                              mean=self.cfg.INPUT.PIXEL_MEAN)

        if self.use_salience and not self.use_parsing:
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')

            if self.transform_salience_parsing == None:
                salience_img = preprocess_salience(read_numpy_file(salience_path))
            else:
                random.seed(seed)
                # salience_img = self.transform_salience_parsing(Image.fromarray(read_numpy_file(salience_path)))
                salience_img = mask_transform(Image.fromarray(read_numpy_file(salience_path)))
                salience_img=MREA(salience_img)
                salience_img=ToPILImage()(salience_img)
                salience_img=T.Resize((64,32))(salience_img)
                salience_img=T.ToTensor()(salience_img)

                # print(salience_img.size())
                # import pdb
                # pdb.set_trace()
                # salience_img = salience_img.resize((32, 64), Image.BILINEAR)
                # salience_img = np.array(salience_img)
                # salience_img =torch.Tensor(salience_img)

            return img, pid, camid, salience_img, img_path
        elif not self.use_salience and self.use_parsing:
            parsing_path = osp.join(self.parsing_base_path, get_image_name(img_path) + '.npy')
            zerovalue=get_value(parsing_path)
            # parsing_img, label_mark = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            parsing_img = new_decode_parsing(torch.tensor(read_numpy_file(parsing_path)),zerovalue)
            body_parsing_img = body_decode_parsing(torch.tensor(read_numpy_file(parsing_path)))

            if self.transform_salience_parsing != None:
                new_parsing_img = []
                body_new_parsing_img = []
                for slide in parsing_img:
                    random.seed(seed)
                    # img_i = self.transform_salience_parsing(Image.fromarray(slide))
                    img_i=mask_transform(Image.fromarray(slide))
                    img_i = MREA(img_i)
                    img_i = ToPILImage()(img_i)
                    img_i = T.Resize((64, 32))(img_i)
                    # img_i = T.ToTensor()(img_i)
                    # img_i = img_i.resize((64, 192), Image.BILINEAR)
                    # img_i = img_i.resize((32, 64), Image.BILINEAR)
                    img_i = np.array(img_i)
                    new_parsing_img.append(img_i)
                for slide in body_parsing_img:
                    random.seed(seed)
                    img_i = self.transform_salience_parsing(Image.fromarray(slide))
                    # img_i = img_i.resize((64, 192), Image.BILINEAR)
                    # img_i = img_i.resize((32, 64), Image.BILINEAR)
                    img_i = np.array(img_i)
                    body_new_parsing_img.append(img_i)

                # parsing_img = np.array(new_parsing_img)
                parsing_img = torch.Tensor(new_parsing_img)
                # import pdb
                # pdb.set_trace()
                # body_new_parsing_img = np.array(body_new_parsing_img)
                body_new_parsing_img = torch.Tensor(body_new_parsing_img)

            # return img, pid, camid, parsing_img,body_new_parsing_img, img_path
            return img, pid, camid, parsing_img, img_path

        elif self.use_parsing and self.use_salience:
            parsing_path = osp.join(self.parsing_base_path, get_image_name(img_path) + '.npy')
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')

            if self.transform_salience_parsing == None:
                salience_img = preprocess_salience(read_numpy_file(salience_path))
                parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            else:
                random.seed(seed)
                salience_img = self.transform_salience(Image.fromarray(read_numpy_file(salience_path)))
                salience_img = salience_img.resize((64, 128), Image.BILINEAR)
                salience_img = np.array(salience_img)

                parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
                new_parsing_img = []
                for slide in parsing_img:
                    random.seed(seed)
                    img_i = self.transform_salience_parsing(Image.fromarray(slide))
                    img_i = img_i.resize((64, 128), Image.BILINEAR)
                    img_i = np.array(img_i)
                    new_parsing_img.append(img_i)
                parsing_img = np.array(new_parsing_img)

            return img, pid, camid, salience_img, parsing_img, img_path
        else:
            return img, pid, camid, img_path
