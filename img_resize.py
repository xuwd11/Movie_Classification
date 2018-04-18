#!/bin/python
#Python3

# so that we will have all images in one folder as jpgs and all labels as well

import argparse
from PIL import Image
import os
import re
from shutil import copyfile
def parse_args():
    parser = argparse.ArgumentParser(description="turn images to SIZExSIZE")
    parser.add_argument('--input',dest="in_img",default="input",type=str)
    parser.add_argument('--output',dest="out_img",default="output",type=str)
    parser.add_argument('--size', dest='size',default=224, type=int)
    args = parser.parse_args()
    return args

def list_files(fldr):
    f_list = []
    for root, directories, filenames in os.walk(fldr):
        for f in filenames:
            f_list.append(os.path.join(root,f))
    return f_list

def save_img(old,new,size):
    Image.open(old).resize((size,size),Image.LANCZOS).save(new)

count = 0
args = parse_args()
if not os.path.exists(args.in_img):
    raise ValueError('No input folder!')
if os.path.exists(args.out_img):
    raise ValueError('Output folder already exist!')
os.makedirs(args.out_img)
for img in list_files(args.in_img):
    new_path = re.sub('^'+args.in_img,args.out_img,img)[:-3] + 'jpg'
    save_img(img,new_path,args.size)
    print('Finished',img)
print('done')
