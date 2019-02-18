import os
import pydicom
import parse
import subprocess
import shlex

def make_dir(target_dir):
    if not os.path.exists(target_dir):
        cmd = "sudo mkdir -p {}".format(target_dir)
        os.system(cmd)
        cmd = 'sudo chmod 777 -R {}'.format(target_dir)
        os.system(cmd)


def decompress(src_dir, dst_dir):
    print '''
    ---------------decompress-------------
    '''

    ## Check target dir
    #     make_dir(dst_dir)

    case_list = os.listdir(src_dir)

    cnt = 0
    a = 0
    b = 0
    for case in case_list:
        full_src_dir = "{}/{}".format(src_dir, case)
        full_dest_dir = "{}".format(dst_dir)

        #         print case

        dcm_list = os.listdir(full_src_dir)
        #         print dcm_list
        PA = []
        AP = []
        LL = []
        for dcm in dcm_list:
            if dcm.endswith(".dcm"):
                full_dcm_filename = "{}/{}".format(full_src_dir, dcm)
                print full_dcm_filename
                ds = pydicom.read_file(full_dcm_filename, force=True)
                print ds.ViewPosition
                dest_name = "{}".format(case)
                print dest_name
                target_dcm_filename = "{}/{}/{}.dcm".format(full_dest_dir, ds.ViewPosition, dest_name)
                print target_dcm_filename
                cmd = "sudo gdcmconv --raw {} {}".format(full_dcm_filename, target_dcm_filename)
                subprocess.call(shlex.split(cmd))
            else:
                full_dcm_filename = "{}/{}".format(full_src_dir, dcm)
                print full_dcm_filename
                ds = pydicom.read_file(full_dcm_filename, force=True)
                print ds.ViewPosition
                dest_name = "{}".format(case)
                print dest_name
                target_dcm_filename = "{}/{}/{}.dcm".format(full_dest_dir, ds.ViewPosition, dest_name)
                print target_dcm_filename
                cmd = "sudo gdcmconv --raw {} {}".format(full_dcm_filename, target_dcm_filename)
                subprocess.call(shlex.split(cmd))

            if not (ds.Modality == "DX" or ds.Modality == "CR"):
                print "Wrong Modaility {}".format(ds.Modality)
                continue

        #             try:
        #                 print ds.ViewPosition
        #             except:
        #                 continue
        #             if ds.ViewPosition == "PA" or ds.ViewPosition == "LL":
        #                 dest_name = "{}".format(case)
        #                 print dest_name
        #                 target_dcm_filename = "{}/{}/{}.dcm".format(full_dest_dir,ds.ViewPosition, dest_name)
        #                 print target_dcm_filename

        # #                 cmd = "sudo gdcmconv --raw {} {}".format(full_dcm_filename, target_dcm_filename)
        # #                 subprocess.call(shlex.split(cmd))
        # #                 print "{} {}".format(cnt,cmd)
        #                 break
        cnt += 1
    print "Total : {}".format(cnt)
    return dst_dir

if __name__ == "__main__":

    # Preprocessing
    src_dir = "/media/samba_share/data/XrayLungSeg/data/cxr/00_MGH_dicom"
    dst_dir = "/media/samba_share/data/Xray_view_detector"
    a= decompress(src_dir,dst_dir)

def get_img_from_DICOM(ds):
    if ds.PixelData == "":
        status = False
        img = np.zeros((10, 10), dtype=np.uint8)
    else:
        status = True
        intercept = ds.RescaleIntercept
        slope = ds.RescaleSlope
        img = ds.pixel_array.copy()
        img = img.astype(np.float32)
        img = slope * img + intercept
    return img

def CLAHE(img, cliplimit=2.0, tilegridsize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tilegridsize)
    img = clahe.apply(img)
    return img

def convert_colorbase(ds, img):
    if str(ds.PhotometricInterpretation) == "MONOCHROME1":
        img = 255 - img
    else:
        img = img
    return img

def resize(img, input_shape, interpolation):
    # resize image after zero padding
    height, width = img.shape
    if height >= width:
        dim_y = height
        dim_x = dim_y
        img_temp = np.zeros((dim_y,dim_x), dtype=np.uint8)
        x_new = (dim_x-width)/2
        y_new = 0
        img_temp[y_new:y_new+height, x_new:x_new+width] = img
    else:
        dim_x = width
        dim_y = dim_x
        img_temp = np.zeros((dim_y,dim_x), dtype=np.uint8)
        x_new = 0
        y_new = (dim_y-height)/2
        img_temp[y_new:y_new+height, x_new:x_new+width] = img

    img_resized = cv2.resize(img_temp, input_shape, interpolation=interpolation)

    return img_resized

def histogram_equalization(img):
    """
        Convert min ~ max range to 0 ~ 255 range
    """

    img = img.astype(np.float64)

    min_val = np.min(img)
    max_val = np.max(img)
    img = img - min_val
    img = img/(max_val-min_val+1e-8)

    img = img*255.0
    img = img.astype(np.uint8)

    return img


import cv2

import parse
# from labeling import *
import subprocess
import shlex

# from TB_deploy import *
# from get_lung_roi import *
import os, sys
import cv2
import pydicom
import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.models import Model, model_from_json
import logging
# from configs import *
from matplotlib import pyplot as plt
import matplotlib


def preprocess(src_dir):
    ## Load dcm list
    dcm_list = [x for x in os.listdir(src_dir) if x.endswith('.dcm')]
    cnt = 0
    print '''
        ---------------preprocess-------------
    '''

    for dcm in dcm_list:
        full_path = "{}/{}".format(src_dir, dcm)

        ## Load pixels
        ds = pydicom.read_file(full_path, force=True)
        print full_path
        img = get_img_from_DICOM(ds)

        img = histogram_equalization(img)
        print np.max(img)
        img = convert_colorbase(ds, img)

        img = CLAHE(img)
        print np.max(img)
        print img.shape
        img = resize(img, (299, 299), cv2.INTER_CUBIC)
        print img.shape
        plt.imshow(img)
        plt.show()
        print np.max(img)
        #         np.save('/media/samba_share/data/XrayLungSeg/data/labels/seg_files/fuse/{}/cxr/{}'.format(dcm[:-4],dcm[:-4]),img)
        cv2.imwrite(('/media/samba_share/data/Xray_view_detector/PNG/LL/{}.png'.format(dcm[:-4], dcm[:-4])), img)
        cnt += 1
    #         noseg_dir = src_dir + "_png"
    #         make_dir(noseg_dir)

    #         img_name = "{}/{}.png".format(noseg_dir, dcm[:-4])
    #         try:
    #             cv2.imwrite(img_name, img)
    #             cnt += 1
    #             print "{} Write a image : {}".format(cnt, img_name)
    #         except:
    #             print "Cannot write a image."

    print "Total : {}".format(cnt)


#     return noseg_dir

if __name__ == "__main__":
    # Preprocessing
    src_dir = "/media/samba_share/data/Xray_view_detector/Dicom_decomp/LL"

    #     dest_dir = decompress(src_dir, dst_dir=dst_dir)
    dest_dir = preprocess(src_dir)