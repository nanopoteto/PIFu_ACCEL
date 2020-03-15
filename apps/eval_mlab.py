import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from apps.train import gen_mesh as gen_mesh
from apps.train_color import gen_mesh as gen_mesh_color

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from torchy.model import *
from torchy.data import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm
import cv2

# get options
opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id)

        # create net
        if opt.netG == 'piximp':
            netG = ConvPiximpNet(opt, projection_mode).to(device=cuda)

        elif opt.netG == 'fanimp':
            netG = FanPiximpNet(opt, projection_mode).to(device=cuda)

        else:
            netG = VhullPiximpNet(opt).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResPixcolNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def load_image(self, image_path, mask_path):
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        # Mask
        mask = Image.open(mask_path).convert('L')
        mask = transforms.Resize(self.load_size)(mask)
        mask = np.tile(np.where(np.array(mask)[:, :]>200, 255, 0)[:, :, None], (1,1,3))
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = transforms.ToTensor()(mask).float()
        # image
        image = Image.open(image_path).convert('RGB')
        image = transforms.Resize(self.load_size)(image)
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            if self.netC:
                gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            else:
                gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)

    def get_bbox(self, msk):
        rows = np.any(msk, axis=1)
        cols = np.any(msk, axis=0)
        rmin, rmax = np.where(rows)[0][[0,-1]]
        cmin, cmax = np.where(cols)[0][[0,-1]]

        return rmin, rmax, cmin, cmax

    def process_img(self, img, msk, bbox=None):
        if msk.shape[0] != msk.shape[1]:
            mask_np = np.array(msk)
            image_np = np.array(img)

            h, w, c = image_np.shape
            pad = h-w
            pad_l = (h-w)//2
            pad_r = pad-pad_l

            image = np.zeros((h, h, 3), dtype=np.uint8)
            mask = np.zeros((h, h), dtype=np.uint8)

            image[:, pad_l:h-pad_r] = image_np
            mask[:, pad_l:h-pad_r] = mask_np

            image_tmp = np.zeros((h*3, h*3, 3), dtype=np.uint8)
            mask_tmp = np.zeros((h*3, h*3), dtype=np.uint8)

            image_tmp[h:h*2, h:h*2] = image
            mask_tmp[h:h*2, h:h*2] = mask

            img = image_tmp
            msk = mask_tmp

        if bbox is None:
            bbox = evaluator.get_bbox(msk > 100)
        # print(bbox)
        
        cx = (bbox[3] + bbox[2])//2
        cy = (bbox[1] + bbox[0])//2

        height = int(1.138*(bbox[1] - bbox[0]))
        hh = height//2

        print('------------')
        print(cy-hh)
        print(cy+hh)
        print('------------')

        img = img[cy-hh:(cy+hh),cx-hh:(cx+hh),:]
        msk = msk[cy-hh:(cy+hh),cx-hh:(cx+hh)]

        # img = cv2.resize(img, (512,512))
        # msk = cv2.resize(msk, (512,512))

        # kernel = np.ones((5,5),np.uint8)
        # msk = cv2.erode((255*(msk > 100)).astype(np.uint8), kernel, iterations = 1)

        return img, msk


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    ''' 
    Args
        input_image (path): An image is not square and have alpha channel.
    '''

    ''' Preprocess

    1. extract mask
    2. align person position

    '''

    # 1. extract mask
    if True:
        test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
        for test_image in test_images:
            img = cv2.imread(test_image, -1)
            print(img.shape)
            print(test_image)
            mask = np.where(img[:, :, 3]>200, 255, 0)

            print(test_image[:-4]+'_mask.png')
            cv2.imwrite(test_image[:-4]+'_mask.png', mask)


    # 2. align person position
    if True:
        test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
        test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
        test_masks = [f[:-4]+'_mask.png' for f in test_images]

        for image_path, mask_path in zip(test_images, test_masks):
            img = cv2.imread(image_path)
            msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img_new, msk_new = evaluator.process_img(img, msk)
            cv2.imwrite(image_path, img_new)
            cv2.imwrite(mask_path, msk_new)
    

    # 3. 3D Reconstruction Process
    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    test_masks = [f[:-4]+'_mask.png' for f in test_images]

    print("num; ", len(test_masks))

    for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
        try:
            print(image_path, mask_path)
            data = evaluator.load_image(image_path, mask_path)
            evaluator.eval(data)
        except Exception as e:
           print("error:", e.args)
