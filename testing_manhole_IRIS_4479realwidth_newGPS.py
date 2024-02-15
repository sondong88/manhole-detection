"""02. Predict with pre-trained Faster RCNN models
==============================================

This article shows how to play with pre-trained Faster RCNN model.

First let's import some necessary libraries:
"""
import sys
import os
import random
import cv2
import numpy as np
import mxnet as mx

from matplotlib import pyplot as plt

from gluoncv.data.transforms.presets.rcnn import transform_test
from PyQt5.QtWidgets import (QMainWindow, QTextEdit, 
    QAction, QFileDialog, QApplication)
import xml.etree.ElementTree as ET

import xlsxwriter
import Calculate_GPS_IRIS_4479realwidth_newGPS
import combine_GPS_manholes

from datetime import date
today = date.today()
import argparse

import time


## set up
# realwidth = 4479 # realwidth of 3D data
manhole_detection = True
save_image_result = False
save_image_manhole_crack_1800mm = True
save_image_manhole_crack_2500mm = True
save_image_manhole_crop = True
save_image_manhole_crop_2500mm = True
save_image_manhole_crop_10000mm = True
save_image_original_4479_10000 = True

## Use GPU if one exists, else use CPU
# ctx = mx.gpu(0) if mx.context.num_gpus() else mx.cpu()
def choose_GPU():
    number_GPU = mx.context.num_gpus()
    if number_GPU == 0: 
        print('There is no GPU device. The program need to GPU for running')
        sys.exit()
    else:
        for GPU_i in range(number_GPU):
            print('GPU:', GPU_i)
            try: 
                GPU_free = min(mx.context.gpu_memory_info(device_id=GPU_i)[0], mx.context.gpu_memory_info(device_id=GPU_i)[0], mx.context.gpu_memory_info(device_id=GPU_i)[0])
                print('GPU_free', GPU_free)
            except: continue
            if  GPU_free > 4800000000:
                # ctx = mx.gpu(GPU_i)
                GPU_name = GPU_i
                break
            if GPU_i == number_GPU -1:
                print('There is no enough GPU device memory > 4800000000')
                sys.exit()
    return GPU_name
GPU_name = choose_GPU()
def parse_args():
    parser = argparse.ArgumentParser(description='manhole program--Dong')
    parser.add_argument('--gpus', type=str, default='',
                        help='Training with GPUs, you can specify 1,3 for example.')
    args = parser.parse_args()
    return args
args = parse_args()
if args.gpus == '': ctx = mx.gpu(GPU_name)
else: ctx = mx.gpu(int(args.gpus))

# print(args.gpus)
# sys.exit()


## Get the current working directory
def get_main_source_dir(root_dir=None, name = 'Open annotations Directory'):
    if root_dir is None:
        root_dir = '/media'
    main_source_dir = (QFileDialog.getExistingDirectory(None,name, root_dir))
    return main_source_dir

######################################################################
# Load a pretrained model

def faster_rcnn_resnet50_v1b_manhole_4479_10000(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_v1b_voc(pretrained=True)
    >>> print(model)
    """
    # from ....model_zoo.resnetv1b import resnet50_v1b
    from gluoncv.model_zoo.resnetv1b import resnet50_v1b
    from mxnet.gluon import nn
    from gluoncv.model_zoo.rcnn.faster_rcnn import get_faster_rcnn
    # from ....data import VOCDetection
    # classes = VOCDetection.CLASSES

    classes = ('manhole_type_0', 'manhole_type_1', 'manhole_type_10', 'manhole_type_11', 'manhole_type_12',
               'manhole_type_13', 'manhole_type_14', 'manhole_type_15', 'manhole_type_16',
               'manhole_type_17', 'manhole_type_18', 'manhole_type_19', 'manhole_type_2', 'manhole_type_20',
               'manhole_type_21', 'manhole_type_22', 'manhole_type_24', 'manhole_type_25',
               'manhole_type_26', 'manhole_type_27', 'manhole_type_29', 'manhole_type_3', 'manhole_type_30',
               'manhole_type_31', 'manhole_type_32', 'manhole_type_33', 'manhole_type_34',
               'manhole_type_35', 'manhole_type_36', 'manhole_type_37', 'manhole_type_38', 'manhole_type_4',
               'manhole_type_5', 'manhole_type_6', 'manhole_type_7', 'manhole_type_8',
               'manhole_type_9', 'no_manhole')
    # classes = ('manhole',)
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet50_v1b', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)

def faster_rcnn_resnet101_v1d_manhole_1600_1600(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_v1b_voc(pretrained=True)
    >>> print(model)
    """
    # from ....model_zoo.resnetv1b import resnet50_v1b
    from gluoncv.model_zoo.resnetv1b import resnet101_v1d
    from mxnet.gluon import nn
    from gluoncv.model_zoo.rcnn.faster_rcnn import get_faster_rcnn
    # from ....data import VOCDetection
    # classes = VOCDetection.CLASSES
    classes = ('manhole',)
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet101_v1d', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)

def faster_rcnn_resnet101_v1d_manhole_1600_1600_detail(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_v1b_voc(pretrained=True)
    >>> print(model)
    """
    # from ....model_zoo.resnetv1b import resnet50_v1b
    from gluoncv.model_zoo.resnetv1b import resnet101_v1d
    from mxnet.gluon import nn
    from gluoncv.model_zoo.rcnn.faster_rcnn import get_faster_rcnn
    # from ....data import VOCDetection
    # classes = VOCDetection.CLASSES
    classes = ('manhole_type_0', 'manhole_type_10_0', 'manhole_type_10_1', 'manhole_type_10_2', 'manhole_type_10_3', 'manhole_type_11_0', 'manhole_type_11_1', 'manhole_type_11_2', 
                'manhole_type_12_1', 'manhole_type_12_2', 'manhole_type_13_0', 'manhole_type_13_1', 'manhole_type_13_2', 'manhole_type_13_3', 'manhole_type_14_1', 'manhole_type_15_0', 
                'manhole_type_15_1', 'manhole_type_15_2', 'manhole_type_15_3', 'manhole_type_16_1', 'manhole_type_17_1', 'manhole_type_18_1', 'manhole_type_19_1', 'manhole_type_19_2', 
                'manhole_type_1_0', 'manhole_type_1_1', 'manhole_type_1_10', 'manhole_type_1_11', 'manhole_type_1_12', 'manhole_type_1_13', 'manhole_type_1_14', 'manhole_type_1_15', 
                'manhole_type_1_16', 'manhole_type_1_17', 'manhole_type_1_18', 'manhole_type_1_19', 'manhole_type_1_2', 'manhole_type_1_20', 'manhole_type_1_21', 'manhole_type_1_22', 
                'manhole_type_1_23', 'manhole_type_1_24', 'manhole_type_1_25', 'manhole_type_1_26', 'manhole_type_1_27', 'manhole_type_1_28', 'manhole_type_1_29', 'manhole_type_1_3', 
                'manhole_type_1_30', 'manhole_type_1_31', 'manhole_type_1_32', 'manhole_type_1_33', 'manhole_type_1_34', 'manhole_type_1_35', 'manhole_type_1_36', 'manhole_type_1_37', 
                'manhole_type_1_38', 'manhole_type_1_39', 'manhole_type_1_4', 'manhole_type_1_40', 'manhole_type_1_41', 'manhole_type_1_42', 'manhole_type_1_43', 'manhole_type_1_44', 
                'manhole_type_1_45', 'manhole_type_1_46', 'manhole_type_1_47', 'manhole_type_1_48', 'manhole_type_1_49', 'manhole_type_1_5', 'manhole_type_1_51', 'manhole_type_1_52', 
                'manhole_type_1_53', 'manhole_type_1_54', 'manhole_type_1_55', 'manhole_type_1_57', 'manhole_type_1_58', 'manhole_type_1_59', 'manhole_type_1_6', 'manhole_type_1_60', 
                'manhole_type_1_61', 'manhole_type_1_62', 'manhole_type_1_63', 'manhole_type_1_7', 'manhole_type_1_8', 'manhole_type_1_9', 'manhole_type_20_1', 'manhole_type_20_2', 
                'manhole_type_20_3', 'manhole_type_20_4', 'manhole_type_20_6', 'manhole_type_21_1', 'manhole_type_22_1', 'manhole_type_24_1', 'manhole_type_24_2', 'manhole_type_25_1', 
                'manhole_type_26_1', 'manhole_type_27_1', 'manhole_type_29_1', 'manhole_type_2_0', 'manhole_type_2_1', 'manhole_type_30_1', 'manhole_type_31_1', 'manhole_type_32_1', 
                'manhole_type_32_2', 'manhole_type_33_1', 'manhole_type_34_1', 'manhole_type_35_1', 'manhole_type_36_1', 'manhole_type_37_1', 'manhole_type_38_0', 'manhole_type_38_1', 
                'manhole_type_39_1', 'manhole_type_3_0', 'manhole_type_3_1', 'manhole_type_3_2', 'manhole_type_3_3', 'manhole_type_3_4', 'manhole_type_3_5', 'manhole_type_3_6', 'manhole_type_3_7', 
                'manhole_type_3_8', 'manhole_type_4_1', 'manhole_type_4_2', 'manhole_type_4_3', 'manhole_type_5_0', 'manhole_type_5_1', 'manhole_type_5_2', 'manhole_type_6_1', 'manhole_type_7_0', 
                'manhole_type_7_1', 'manhole_type_7_2', 'manhole_type_8_1', 'manhole_type_8_2', 'manhole_type_9_1', 'manhole_type_9_2')
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet101_v1d', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)

def faster_rcnn_resnet101_v1d_manhole_crack_800_800(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_v1b_voc(pretrained=True)
    >>> print(model)
    """
    # from ....model_zoo.resnetv1b import resnet50_v1b
    from gluoncv.model_zoo.resnetv1b import resnet101_v1d
    from mxnet.gluon import nn
    from gluoncv.model_zoo.rcnn.faster_rcnn import get_faster_rcnn
    # from ....data import VOCDetection
    # classes = VOCDetection.CLASSES
    classes = ('fa_crk_hi', 'fa_crk_lo', 'linear_crk_hi', 'linear_crk_lo')

    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet101_v1d', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)

# *******Dong*************************************************
# net_4479_10000 = faster_rcnn_resnet50_v1b_manhole_4479_10000(pretrained=False, pretrained_base=True, ctx=ctx)
# net_4479_10000.load_parameters('/media/iris/EA18A64718A6129D/Dong/back up/Dong_/Dong_manhole/manhole_test_20210321/params_model/faster_rcnn_resnet50_v1b_voc_4479_10000_basicname_nomanhole_2.params', ctx=ctx, ignore_extra=True)
# # net_4479_10000.load_parameters(os.path.join(current_dir, "params_model", "faster_rcnn_resnet50_v1b_voc_4479_10000_basicname_nomanhole_2.params"), ctx=ctx)

# net_1600_1600 = faster_rcnn_resnet101_v1d_manhole_1600_1600(pretrained=False, pretrained_base=True, ctx=ctx)
# net_1600_1600.load_parameters('/media/iris/EA18A64718A6129D/Dong/back up/Dong_/Dong_manhole/manhole_test_20210321/params_model/faster_rcnn_resnet101_v1d_voc_onlymanhole_1600_1600_1.params', ctx=ctx, ignore_extra=True)
# # net_1600_1600.load_parameters(os.path.join(current_dir, "params_model", "faster_rcnn_resnet101_v1d_voc_onlymanhole_1600_1600_1.params"), ctx=ctx)

# net_1600_1600_detail = faster_rcnn_resnet101_v1d_manhole_1600_1600_detail(pretrained=False, pretrained_base=True, ctx=ctx)
# net_1600_1600_detail.load_parameters('/media/iris/EA18A64718A6129D/Dong/back up/Dong_/Dong_manhole/manhole_test_20210321/params_model/faster_rcnn_resnet101_v1d_voc_1600_1600_detail_1.params', ctx=ctx, ignore_extra=True)
# # net_1600_1600_detail.load_parameters(os.path.join(current_dir, "params_model", "faster_rcnn_resnet101_v1d_voc_1600_1600_detail_1.params"), ctx=ctx)

# net_800_800_crack = faster_rcnn_resnet101_v1d_manhole_crack_800_800(pretrained=False, pretrained_base=True, ctx=ctx)
# net_800_800_crack.load_parameters('/media/iris/EA18A64718A6129D/Dong/back up/Dong_/Dong_manhole/manhole_test_20210321/params_model/faster_rcnn_resnet101_v1d_voc_manhole_crack_2.params', ctx=ctx, ignore_extra=True)
# # net_800_800_crack.load_parameters(os.path.join(current_dir, "params_model", "faster_rcnn_resnet101_v1d_voc_manhole_crack_2.params"), ctx=ctx)
# *******Dong*************************************************

#*******edit*************************************************
net_4479_10000 = faster_rcnn_resnet50_v1b_manhole_4479_10000(pretrained=False, pretrained_base=True, ctx=ctx)
net_4479_10000.load_parameters('params_model/faster_rcnn_resnet50_v1b_voc_4479_10000_basicname_nomanhole_2.params', ctx=ctx, ignore_extra=True)
# net_4479_10000.load_parameters(os.path.join(current_dir, "params_model", "faster_rcnn_resnet50_v1b_voc_4479_10000_basicname_nomanhole_2.params"), ctx=ctx)

net_1600_1600 = faster_rcnn_resnet101_v1d_manhole_1600_1600(pretrained=False, pretrained_base=True, ctx=ctx)
net_1600_1600.load_parameters('params_model/faster_rcnn_resnet101_v1d_voc_onlymanhole_1600_1600_1.params', ctx=ctx, ignore_extra=True)
# net_1600_1600.load_parameters(os.path.join(current_dir, "params_model", "faster_rcnn_resnet101_v1d_voc_onlymanhole_1600_1600_1.params"), ctx=ctx)

net_1600_1600_detail = faster_rcnn_resnet101_v1d_manhole_1600_1600_detail(pretrained=False, pretrained_base=True, ctx=ctx)
net_1600_1600_detail.load_parameters('params_model/faster_rcnn_resnet101_v1d_voc_1600_1600_detail_1.params', ctx=ctx, ignore_extra=True)
# net_1600_1600_detail.load_parameters(os.path.join(current_dir, "params_model", "faster_rcnn_resnet101_v1d_voc_1600_1600_detail_1.params"), ctx=ctx)

net_800_800_crack = faster_rcnn_resnet101_v1d_manhole_crack_800_800(pretrained=False, pretrained_base=True, ctx=ctx)
net_800_800_crack.load_parameters('params_model/faster_rcnn_resnet101_v1d_voc_manhole_crack_2.params', ctx=ctx, ignore_extra=True)
# net_800_800_crack.load_parameters(os.path.join(current_dir, "params_model", "faster_rcnn_resnet101_v1d_voc_manhole_crack_2.params"), ctx=ctx)
#*******edit*************************************************

print("finished loading parameter of network")
######################################################################


class classes_manhole:

    water_works = ['manhole_type_1_1', 'manhole_type_2_1', 'manhole_type_1_2', 'manhole_type_13_1', 'manhole_type_13_2', 'manhole_type_18_1', 'manhole_type_13_3', 'manhole_type_14_1', 
            'manhole_type_21_1', 'manhole_type_1_3', 'manhole_type_1_4', 'manhole_type_1_5', 'manhole_type_1_6', 'manhole_type 14', 'manhole_type_1_8', 'manhole_type_1_9', 
            'manhole_type_1_10', 'manhole_type_1_11', 'manhole_type_1_12', 'manhole_type_1_13', 'manhole_type_1_14', 'manhole_type_22_1', 'manhole_type_25_1', 'manhole_type_1_52', 
            'manhole_type_1_54', 'manhole_type_1_55', 'manhole_type_1_61', 'manhole_type_38_1']

    sewerage = ['manhole_type_3_1', 'manhole_type_4_1', 'manhole_type_5_1', 'manhole_type_3_2', 'manhole_type_12_1', 'manhole_type_17_1', 'manhole_type_19_1', 'manhole_type_19_2', 
                'manhole_type_3_3', 'manhole_type_3_4', 'manhole_type_4_2', 'manhole_type_3_5', 'manhole_type_4_3', 'manhole_type_12_2', 'manhole_type_3_6', 'manhole_type_3_7', 
                'manhole_type_1_49', 'manhole_type_1_50', 'manhole_type_1_51', 'manhole_type_5_2', 'manhole_type_33_1', 'manhole_type_1_59', 'manhole_type_1_60']

    hydrant = ['manhole_type_6_1', 'manhole_type_7_1', 'manhole_type_8_1', 'manhole_type_8_2', 'manhole_type_7_2']

    electric = ['manhole_type_1_15', 'manhole_type_1_16', 'manhole_type_20_3', 'manhole_type_1_17', 'manhole_type_1_18', 'manhole_type_10_1', 'manhole_type_1_19', 'manhole_type_10_3', 'manhole_type_32_2', 'manhole_type_11_2']

    telecommunication = ['manhole_type_1_20', 'manhole_type_1_21', 'manhole_type_1_22', 'manhole_type_1_23', 'manhole_type_1_24', 'manhole_type_1_25', 'manhole_type_1_26', 
                        'manhole_type_1_27', 'manhole_type_1_28', 'manhole_type_1_29', 'manhole_type_20_1', 'manhole_type_11_1', 'manhole_type_1_30', 'manhole_type_1_31', 
                        'manhole_type_1_32', 'manhole_type_1_33', 'manhole_type_20_2', 'manhole_type_1_34', 'manhole_type_1_35', 'manhole_type_1_36', 'manhole_type_1_47', 
                        'manhole_type_32_1', 'manhole_type_1_57', 'manhole_type_1_62', 'manhole_type_20_4', 'manhole_type_20_5', 'manhole_type_37_1', 'manhole_type_1_63', 'manhole_type_20_6']

    gas = ['manhole_type_9_1', 'manhole_type_9_2', 'manhole_type_15_1', 'manhole_type_15_2', 'manhole_type_16_1', 'manhole_type_1_37', 'manhole_type_24_1', 'manhole_type_26_1', 'manhole_type_15_3', 
            'manhole_type_29_1', 'manhole_type_39_1', 'manhole_type_24_2']

    heat = ['manhole_type_1_38', 'manhole_type_30_1', 'manhole_type_34_1']

    subway = ['manhole_type_3_8', 'manhole_type_1_39', 'manhole_type_1_58']

    cable = ['manhole_type_1_40', 'manhole_type_1_41', 'manhole_type_1_42', 'manhole_type_1_48', 'manhole_type_1_63']

    etc = ['manhole_type_10_2', 'manhole_type_10_3', 'manhole_type_1_43', 'manhole_type_1_44', 'manhole_type_1_45', 'manhole_type_1_46', 'manhole_type_27_1', 'manhole_type_31_1', 
            'manhole_type_0', 'manhole_type_35_1', 'manhole_type_36_1']



def resize_short_within(src, short, max_size, mult_base=1, interp=2):
    """Resizes shorter edge to size but make sure it's capped at maximum size.
    Note: `resize_short_within` uses OpenCV (not the CV2 Python library).
    MXNet must have been built with OpenCV for `resize_short_within` to work.
    Resizes the original image by setting the shorter edge to size
    and setting the longer edge accordingly. Also this function will ensure
    the new image will not exceed ``max_size`` even at the longer side.
    Resizing function is called from OpenCV.

    Parameters
    ----------
    src : NDArray
        The original image.
    short : int
        Resize shorter side to ``short``.
    max_size : int
        Make sure the longer side of new image is smaller than ``max_size``.
    mult_base : int, default is 1
        Width and height are rounded to multiples of `mult_base`.
    interp : int, optional, default=2
        Interpolation method used for resizing the image.
        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method mentioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    Returns
    -------
    NDArray
        An 'NDArray' containing the resized image.
    Example
    -------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> new_image = resize_short_within(image, short=800, max_size=1000)
    >>> new_image
    <NDArray 667x1000x3 @cpu(0)>
    >>> new_image = resize_short_within(image, short=800, max_size=1200)
    >>> new_image
    <NDArray 800x1200x3 @cpu(0)>
    >>> new_image = resize_short_within(image, short=800, max_size=1200, mult_base=32)
    >>> new_image
    <NDArray 800x1184x3 @cpu(0)>
    """
    from mxnet.image.image import _get_interp_method as get_interp
    h, w, _ = src.shape
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
        # fit in max_size
        scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
    new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base),
                    int(np.round(h * scale / mult_base) * mult_base))
    return mx.image.imresize(src, new_w, new_h, interp=get_interp(interp, (h, w, new_h, new_w)))


def transform_test(imgs, real_width, real_height, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        orig_img = mx.image.imresize(img, real_width, real_height, interp= "1")
        # orig_img = img.asnumpy().astype('uint8')
        img = resize_short_within(orig_img, short, max_size)
        # orig_img = mx.image.imresize(img, real_width, real_height, interp= "1")
        # orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def load_test(filenames, real_width, real_height, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    imgs = [mx.image.imread(f) for f in filenames]
    # for bad images
    # imgs = cv2.imread(filenames,0)
    # imgs = cv2.equalizeHist(imgs)
    # imgs = mx.nd.array([imgs, imgs, imgs])
    # imgs = mx.nd.transpose(imgs, (1,2,0))
    return transform_test(imgs, real_width, real_height, short, max_size, mean, std)


def make_folder(save_folder_path):
    save_imgs_folder_path = os.path.join(save_folder_path, "image")
    save_annotations_folder_path = os.path.join(save_folder_path, "annotations")
    save_excel_folder_path = os.path.join(save_folder_path, "excel")
    save_imgs_original_folder_path = os.path.join(save_folder_path, "image_original")
    save_crop_manhole_folder_path = os.path.join(save_folder_path, "crop_manhole")
    save_imgs_check_folder_path = os.path.join(save_folder_path, "image_check_again")
    save_crack_manhole_folder_path = os.path.join(save_folder_path, "crack_manhole")
    save_crop_manhole_250mm_folder_path = os.path.join(save_folder_path, "crop_manhole_250mm")
    save_crop_manhole_1000mm_folder_path = os.path.join(save_folder_path, "crop_manhole_1000mm")
    if not os.path.exists(save_imgs_folder_path):
        os.makedirs(save_imgs_folder_path)
    if not os.path.exists(save_annotations_folder_path):
        os.makedirs(save_annotations_folder_path)
    if not os.path.exists(save_excel_folder_path):
        os.makedirs(save_excel_folder_path)
    if not os.path.exists(save_imgs_original_folder_path):
        os.makedirs(save_imgs_original_folder_path)
    if not os.path.exists(save_crop_manhole_folder_path):
        os.makedirs(save_crop_manhole_folder_path)
    if not os.path.exists(save_imgs_check_folder_path):
        os.makedirs(save_imgs_check_folder_path)
    if not os.path.exists(save_crack_manhole_folder_path):
        os.makedirs(save_crack_manhole_folder_path)
    if not os.path.exists(save_crop_manhole_250mm_folder_path):
        os.makedirs(save_crop_manhole_250mm_folder_path)
    if not os.path.exists(save_crop_manhole_1000mm_folder_path):
        os.makedirs(save_crop_manhole_1000mm_folder_path)
    return save_imgs_folder_path, save_annotations_folder_path, save_excel_folder_path, save_imgs_original_folder_path, save_crop_manhole_folder_path, save_imgs_check_folder_path, save_crack_manhole_folder_path, save_crop_manhole_250mm_folder_path, save_crop_manhole_1000mm_folder_path

def get_bbox_4479_10000(img, input_img, bboxes, scores=None, labels=None, thresh=0.5,
                 class_names=None, absolute_coordinates=True):
    """Visualize bounding boxes with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).
    scale : float
        The scale of output image, which may affect the positions of boxes
    linewidth : int, optional, default 2
        Line thickness for bounding boxes.
        Use negative values to fill the bounding boxes.
    """


    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    # if isinstance(img, mx.nd.NDArray):
        # img = img.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if len(bboxes) < 1:
        return img

    if absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]/input_img.shape[2]
        width = img.shape[1]/input_img.shape[3]

        bboxes[:, (0, 2)] *= height
        bboxes[:, (1, 3)] *= width


    result_manhole_4479_10000 = []
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        
        cls_id = int(labels.flat[i]) if labels is not None else -1

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''

        # bbox_new.append(bbox)
        # class_name_new.append(class_name)
        # score_new.append(score)
        if class_name != 'no_manhole': result_manhole_4479_10000.append([class_name, score, bbox])
        # if class_name or score:
        #     y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        #     cv2.putText(img, '{:s} {:s}'.format(class_name, score),
        #                 (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(scale/2, 2),
        #                 bcolor, min(int(scale), 5), lineType=cv2.LINE_AA)
    # print(class_name_new)
    # print(score_new)
    # # print(bbox_new)
    print(result_manhole_4479_10000)
    return result_manhole_4479_10000


def get_bbox_1600_1600(img, input_img, bboxes, scores=None, labels=None, thresh=0.5,
                 class_names=None, colors=None,  classes_manhole = classes_manhole(), 
                 absolute_coordinates=True, scale=1.0, linewidth=10, x_crop_min = 0, y_crop_min = 0):

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if len(bboxes) < 1:
        return img

    if absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = 1600.0/input_img.shape[2]
        width = 1600.0/input_img.shape[3]
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)]*height + x_crop_min
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)]*width + y_crop_min
    else:
        bboxes *= scale


    # use random colors if None is provided
    if colors is None:
        colors = dict()

    result_manhole_1600_1600 = []
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        
        cls_id = int(labels.flat[i]) if labels is not None else -1

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''

        result_manhole_1600_1600.append([class_name, score, bbox])

    # print(result_manhole_1600_1600)
    return result_manhole_1600_1600

def get_bbox_800_800(img, input_img, bboxes, scores=None, labels=None, thresh=0.5,
                 class_names=None, colors=None,  classes_manhole = classes_manhole(), 
                 absolute_coordinates=True, scale=1.0, linewidth=10, x_crop_min = 0, y_crop_min = 0):

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if len(bboxes) < 1:
        return img

    if absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = 800.0/input_img.shape[2]
        width = 800.0/input_img.shape[3]
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)]*height + x_crop_min
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)]*width + y_crop_min
    else:
        bboxes *= scale


    # use random colors if None is provided
    if colors is None:
        colors = dict()

    result_manhole_1600_1600 = []
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        
        cls_id = int(labels.flat[i]) if labels is not None else -1

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''

        result_manhole_1600_1600.append([class_name, score, bbox])

    # print(result_manhole_1600_1600)
    return result_manhole_1600_1600

def draw_result_image(img, result_manhole_1600_1600_i):
    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    [class_name, score, bbox] = result_manhole_1600_1600_i
    xmin, ymin, xmax, ymax = [int(x) for x in bbox]

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 5)

    if class_name in classes_manhole.water_works: class_name = 'water_works'
    elif class_name in classes_manhole.sewerage: class_name = 'sewerage'
    elif class_name in classes_manhole.hydrant: class_name = 'hydrant'
    elif class_name in classes_manhole.electric: class_name = 'electric'
    elif class_name in classes_manhole.telecommunication: class_name = 'telecommunication'
    elif class_name in classes_manhole.gas: class_name = 'gas'
    elif class_name in classes_manhole.heat: class_name = 'heat'
    elif class_name in classes_manhole.subway: class_name = 'subway'
    elif class_name in classes_manhole.cable: class_name = 'cable'
    else: class_name = 'etc'

    if class_name or score:
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(img, '{:s} {:s}'.format(class_name, score),
                    (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0,0,255), 4, lineType=cv2.LINE_AA)
    return img

def draw_result_crack(img, result_crack):
    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    [class_name, score, bbox] = result_crack
    xmin, ymin, xmax, ymax = [int(x) for x in bbox]

    if class_name in ('fa_crk_hi', 'fa_crk_lo'): cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
    elif class_name in ('linear_crk_hi', 'linear_crk_lo'): cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)

    return img

def calculate_crack_area(result_crack):
    area = 0
    for result_crack_i in result_crack:
        [class_name, score, bbox] = result_crack_i
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        if   class_name == 'linear_crk_lo': area += (((xmax-xmin)**2 + (ymax-ymin)**2)**(1/2))*20
        elif class_name == 'linear_crk_hi': area += (((xmax-xmin)**2 + (ymax-ymin)**2)**(1/2))*40
        elif class_name == 'fa_crk_lo':     area += (((xmax-xmin)**2 + (ymax-ymin)**2)**(1/2))*40 ##(xmax - xmin)*(ymax - ymin)*0.4
        elif class_name == 'fa_crk_hi':     area += (((xmax-xmin)**2 + (ymax-ymin)**2)**(1/2))*60 ##(xmax - xmin)*(ymax - ymin)*0.8
    return area

def crack_serverity(crack_percent):
    if crack_percent < 5: serverity_level = 'A'
    elif crack_percent < 10: serverity_level = 'B'
    elif crack_percent < 20: serverity_level = 'C'
    else: serverity_level = 'D'
    return serverity_level 

def change_name(class_name):
    if class_name in classes_manhole.water_works: class_name = 'water_works'
    elif class_name in classes_manhole.sewerage: class_name = 'sewerage'
    elif class_name in classes_manhole.hydrant: class_name = 'hydrant'
    elif class_name in classes_manhole.electric: class_name = 'electric'
    elif class_name in classes_manhole.telecommunication: class_name = 'telecommunication'
    elif class_name in classes_manhole.gas: class_name = 'gas'
    elif class_name in classes_manhole.heat: class_name = 'heat'
    elif class_name in classes_manhole.subway: class_name = 'subway'
    elif class_name in classes_manhole.cable: class_name = 'cable'
    else: class_name = 'etc'

    return class_name

def get_bbox_1600_1600_crack(img, input_img, bboxes, scores=None, labels=None, thresh=0.5,
                 class_names=None, colors=None,
                 absolute_coordinates=True, scale=1.0, linewidth=10, x_crop_min = 0, y_crop_min = 0):
    """Visualize bounding boxes with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.d
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).
    scale : float
        The scale of output image, which may affect the positions of boxes
    linewidth : int, optional, default 2
        Line thickness for bounding boxes.
        Use negative values to fill the bounding boxes.

    Returns
    -------
    numpy.ndarray
        The image with detected results.

    """
    # from matplotlib import pyplot as plt
    # from ..filesystem import try_import_cv2
    # cv2 = try_import_cv2()

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if len(bboxes) < 1:
        return img

    if absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = 1000.0/input_img.shape[2]
        width = 1520.0/input_img.shape[3]
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)]*height + x_crop_min
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)]*width + y_crop_min
    else:
        bboxes *= scale


    # use random colors if None is provided
    if colors is None:
        colors = dict()

    ori_image = img.copy()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''

        if class_name in ['trs_crk', 'lg_crk', 'fa_crk']:
            xmin, ymin, xmax, ymax = [int(x) for x in bbox]
            bcolor = [x * 255 for x in colors[cls_id]]
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bcolor, linewidth)
            pts_draw = np.array([[xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax], [xmin-200, ymax]],np.int32)
            pts_draw = pts_draw.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts_draw], bcolor)

            if class_names is not None and cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = str(cls_id) if cls_id >= 0 else ''
            score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''
            if class_name or score:
                y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                cv2.putText(img, '{:s} {:s}'.format(class_name, score),
                            (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(scale/2, 2),
                            bcolor, min(int(scale), 5), lineType=cv2.LINE_AA)
            # print(class_name)
    cv2.addWeighted(img, 0.2, ori_image, 0.8, 0, img)
    return img


def GenerateXML(path, filename, width, height, name_manhole_list, bboxes_list): 
      
    root = ET.Element("annotation") 
      
    root_filename = ET.SubElement(root, "filename") 
    root_filename.text = filename

    size = ET.SubElement(root, "size") 
    size_width = ET.SubElement(size, "width")
    size_width.text = str(width)
    size_height = ET.SubElement(size, "height")
    size_height.text = str(height)

    for i, name_manhole in enumerate(name_manhole_list):
        xmin, ymin, xmax, ymax = [int(x) for x in bboxes_list[i]]
        object1 = ET.SubElement(root, "object")
        object1_name = ET.SubElement(object1, "name")
        object1_name.text = name_manhole
        object1_bndbox = ET.SubElement(object1, "bndbox")
        bndbox_xmin = ET.SubElement(object1_bndbox, "xmin")
        bndbox_ymin = ET.SubElement(object1_bndbox, "ymin")
        bndbox_xmax = ET.SubElement(object1_bndbox, "xmax")
        bndbox_ymax = ET.SubElement(object1_bndbox, "ymax")
        bndbox_xmin.text = str(xmin)
        bndbox_ymin.text = str(ymin)
        bndbox_xmax.text = str(xmax)
        bndbox_ymax.text = str(ymax)
      
    tree = ET.ElementTree(root) 
      
    with open (path, "wb") as files : 
        tree.write(files) 

def isInside(circle_x, circle_y, rad, x, y): 
      
    # Compare radius of circle 
    # with distance of its center 
    # from given point 
    if ((x - circle_x) * (x - circle_x) + 
        (y - circle_y) * (y - circle_y) <= rad * rad): 
        return True; 
    else: 
        return False;

def check_overlap_prediction(result_manhole):
    from itertools import combinations
    result_manhole_comb = combinations(result_manhole, 2)
    for (manhole_A, manhole_B) in  result_manhole_comb:
        boxA, boxB = manhole_A[2], manhole_B[2]
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        # iou = interArea / float(boxAArea + boxBArea - interArea)
        if interArea / float(boxAArea) > 0.5 or interArea / float(boxBArea) > 0.5:
            # result_manhole.remove(manhole_B) if boxAArea > boxBArea else result_manhole.remove(manhole_A)
            if boxAArea > boxBArea:
                try: result_manhole.remove(manhole_B)
                except: print('need to check again')  
            else:
                try: result_manhole.remove(manhole_A)
                except: print('need to check again')
            # bboxes_new.append(manhole_A) if boxAArea > boxBArea else bboxes_new.append(manhole_B)
        # else: 
        #     bboxes_new.append(boxA)
        #     bboxes_new.append(boxB)

    # return the intersection over union value
    return result_manhole

def check_overlap_prediction_classification(result_manhole):
    from itertools import combinations
    result_manhole_comb = combinations(result_manhole, 2)
    for (manhole_A, manhole_B) in  result_manhole_comb:
        boxA, boxB = manhole_A[2], manhole_B[2]
        score_boxA, score_boxB = manhole_A[1], manhole_B[1]
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        # iou = interArea / float(boxAArea + boxBArea - interArea)
        if interArea / float(boxAArea) > 0.5 or interArea / float(boxBArea) > 0.5:
            if float(score_boxA[:-1]) > float(score_boxB[:-1]):
                try: result_manhole.remove(manhole_B)
                except: print('need to check again')  
            else:
                try: result_manhole.remove(manhole_A)
                except: print('need to check again')
            # bboxes_new.append(manhole_A) if boxAArea > boxBArea else bboxes_new.append(manhole_B)
        # else: 
        #     bboxes_new.append(boxA)
        #     bboxes_new.append(boxB)

    # return the intersection over union value
    return result_manhole

## detecting manhole crack 250mm
def detect_manhole_crack(manhole_crop_crack, save_crack_manhole_folder_path, fname, pavement_area, x_center, y_center, result_manhole_1600_1600_i, write_image = True):
    manhole_crack_img = manhole_crop_crack.copy()
    area_crack = 0 
    for row_step in range(2):
        for col_step in range(2):
            manhole_crack = manhole_crop_crack[row_step*800:(row_step+1)*800, col_step*800:(col_step+1)*800]
            manhole_crack, manhole_crack_original = transform_test(mx.nd.array(manhole_crack), real_width = 800, real_height = 800)
            # print(manhole_crack.shape)
            box_ids_crack, scores_crack, bboxes_crack = net_800_800_crack(manhole_crack.as_in_context(ctx))
            result_crack = get_bbox_800_800(manhole_crop_crack, manhole_crack, bboxes_crack[0], scores_crack[0], 
                                                                    box_ids_crack[0], class_names=net_800_800_crack.classes, thresh=0.5, absolute_coordinates=True,
                                                                    x_crop_min = col_step*800, y_crop_min = row_step*800)
            result_crack = check_overlap_crack_prediction(result_crack)
            for result_crack_i in result_crack: manhole_crack_img = draw_result_crack(manhole_crack_img, result_crack_i)
            area_crack += calculate_crack_area(result_crack)
    img_output_path_crack = os.path.join(save_crack_manhole_folder_path, fname.split('.')[0] + '_{}_{}_{}_crack.jpg'.format(change_name(result_manhole_1600_1600_i), x_center, y_center))
    crack_percent = round(area_crack/pavement_area*100,2)
    if write_image == True:
        cv2.putText(manhole_crack_img, 'area: {} %'.format(crack_percent),
                                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0,0,255), 1, lineType=cv2.LINE_AA)
        cv2.imwrite(img_output_path_crack, manhole_crack_img)
    return crack_percent


def check_overlap_prediction_1600_1600(result_manhole_1600_1600, result_manhole_4479_10000):
    result_manhole_1600_1600_copy = []
    for manhole_A in  result_manhole_1600_1600:
        boxA, boxB = manhole_A[2], result_manhole_4479_10000[2]
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # print(iou)
        if iou > 0.5:
            result_manhole_1600_1600_copy.append(manhole_A)
            # bboxes_new.append(manhole_A) if boxAArea > boxBArea else bboxes_new.append(manhole_B)
        # else: 
        #     bboxes_new.append(boxA)
        #     bboxes_new.append(boxB)

    # return the intersection over union value
    return result_manhole_1600_1600_copy

def check_overlap_crack_prediction(result_crack):
    result_crack_new = []
    for i, crack_A in enumerate(result_crack):
        for j, crack_B in enumerate(result_crack):
            if i == j: continue
            boxA, boxB = crack_A[2], crack_B[2]
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            if [xA, yA, xB, yB] == [cor for cor in boxA]:
                result_crack[i][2] = [0,0,0,0]
                continue
            elif  [xA, yA, xB, yB] == [cor for cor in boxB]:
                result_crack[j][2] = [0,0,0,0]
                continue
            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            if interArea > 0:
                if boxA[0] < boxB[0] and boxA[3] < boxB[3]:
                    [xminA, yminA, xmaxA, ymaxA] = boxA
                    [xminB, yminB, xmaxB, ymaxB] = boxB
                    boxB_new_1 = [xmaxA, yminB, xmaxB, ymaxB]
                    boxB_new_2 = [xminB, ymaxA, xmaxB, ymaxB]
                    boxB_new_1Area = (boxB_new_1[2] - boxB_new_1[0] + 1) * (boxB_new_1[3] - boxB_new_1[1] + 1)
                    boxB_new_2Area = (boxB_new_2[2] - boxB_new_2[0] + 1) * (boxB_new_2[3] - boxB_new_2[1] + 1)
                    if boxB_new_1Area > boxB_new_2Area:
                        crack_B[2] = [xmaxA, yminB, xmaxB, ymaxB]
                    else:
                        crack_B[2] = [xminB, ymaxA, xmaxB, ymaxB]
                    result_crack[j] = crack_B
                elif boxA[0] > boxB[0] and boxA[3] > boxB[3]:
                    [xminA, yminA, xmaxA, ymaxA] = boxB
                    [xminB, yminB, xmaxB, ymaxB] = boxA
                    boxA_new_1 = [xmaxA, yminB, xmaxB, ymaxB]
                    boxA_new_2 = [xminB, ymaxA, xmaxB, ymaxB]
                    boxA_new_1Area = (boxA_new_1[2] - boxA_new_1[0] + 1) * (boxA_new_1[3] - boxA_new_1[1] + 1)
                    boxA_new_2Area = (boxA_new_2[2] - boxA_new_2[0] + 1) * (boxA_new_2[3] - boxA_new_2[1] + 1)
                    if boxA_new_1Area > boxA_new_2Area:
                        crack_A[2] = [xmaxA, yminB, xmaxB, ymaxB]
                    else:
                        crack_A[2] = [xminB, ymaxA, xmaxB, ymaxB]
                    result_crack[i] = crack_A
                elif boxA[0] < boxB[0] and boxA[3] > boxB[3]:
                    [xminA, yminA, xmaxA, ymaxA] = boxA
                    [xminB, yminB, xmaxB, ymaxB] = boxB
                    boxB_new_1 = [xminB, yminB, xmaxB, yminA]
                    boxB_new_2 = [xmaxA, yminB, xmaxB, ymaxB]
                    boxB_new_1Area = (boxB_new_1[2] - boxB_new_1[0] + 1) * (boxB_new_1[3] - boxB_new_1[1] + 1)
                    boxB_new_2Area = (boxB_new_2[2] - boxB_new_2[0] + 1) * (boxB_new_2[3] - boxB_new_2[1] + 1)
                    if boxB_new_1Area > boxB_new_2Area:
                        crack_B[2] = boxB_new_1
                    else:
                        crack_B[2] = boxB_new_2
                    result_crack[j] = crack_B
                elif boxA[0] > boxB[0] and boxA[3] < boxB[3]:
                    [xminA, yminA, xmaxA, ymaxA] = boxB
                    [xminB, yminB, xmaxB, ymaxB] = boxA
                    boxA_new_1 = [xminB, yminB, xmaxB, yminA]
                    boxA_new_2 = [xmaxA, yminB, xmaxB, ymaxB]
                    boxA_new_1Area = (boxA_new_1[2] - boxA_new_1[0] + 1) * (boxA_new_1[3] - boxA_new_1[1] + 1)
                    boxA_new_2Area = (boxA_new_2[2] - boxA_new_2[0] + 1) * (boxA_new_2[3] - boxA_new_2[1] + 1)
                    if boxA_new_1Area > boxA_new_2Area:
                        crack_A[2] = boxA_new_1
                    else:
                        crack_A[2] = boxA_new_2
                    result_crack[i] = crack_A 

    for i, crack_A in enumerate(result_crack):
        if [cor for cor in crack_A[2]] != [0,0,0,0]: result_crack_new.append(crack_A)  
    return result_crack_new

def find_center_radius_circle(x1, y1, x2, y2, x3, y3):
    # print("Input three coordinate of the circle:")
    # x1, y1, x2, y2, x3, y3 = map(float, input().split())
    c = (x1-x2)**2 + (y1-y2)**2
    a = (x2-x3)**2 + (y2-y3)**2
    b = (x3-x1)**2 + (y3-y1)**2
    s = 2*(a*b + b*c + c*a) - (a*a + b*b + c*c) 
    px = (a*(b+c-a)*x1 + b*(c+a-b)*x2 + c*(a+b-c)*x3) / s
    py = (a*(b+c-a)*y1 + b*(c+a-b)*y2 + c*(a+b-c)*y3) / s 
    ar = a**0.5
    br = b**0.5
    cr = c**0.5 
    r = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5
    return r, px, py

def crop_manhole_for_calculate_crack(img, xmin, ymin, xmax, ymax):
    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    ## crop image 1600*1600
    x_crop_min = (xmin+xmax)/2 - 800
    x_crop_max = (xmin+xmax)/2 + 800
    y_crop_min = (ymin+ymax)/2 - 800
    y_crop_max = (ymin+ymax)/2 + 800

    if x_crop_min < 0: 
      x_crop_min = 0
      x_crop_max = 1600

    if x_crop_max > 4478: 
      x_crop_max = 4478
      x_crop_min = x_crop_max - 1600

    if y_crop_min < 0:
      y_crop_min = 0
      y_crop_max = 1600 

    if y_crop_max > 9999:
      y_crop_max = 9999
      y_crop_min = y_crop_max - 1600

    crop_img = img[int(y_crop_min):int(y_crop_max), int(x_crop_min):int(x_crop_max)]
    crop_img_1 = crop_img.copy()

    ## crop image from the edge of the manhole is 250mm
    size_box = max((xmax-xmin), (ymax-ymin)) + 500
    x_crop_min_ori = xmin - 250
    x_crop_max_ori = xmax + 250
    y_crop_min_ori = ymin - 250
    y_crop_max_ori = ymax + 250
    
    if x_crop_min_ori < 0: 
      x_crop_min_ori = 0
      x_crop_max_ori = x_crop_min_ori + size_box

    if x_crop_max_ori > 4478: 
      x_crop_max_ori = 4478
      x_crop_min_ori = x_crop_max_ori - size_box

    if y_crop_min_ori < 0:
      y_crop_min_ori = 0
      y_crop_max_ori = y_crop_min_ori + size_box

    if y_crop_max_ori > 9999:
      y_crop_max_ori = 9999
      y_crop_min_ori = y_crop_max_ori - size_box

    crop_manhole_250mm = img[int(y_crop_min_ori):int(y_crop_max_ori), int(x_crop_min_ori):int(x_crop_max_ori)]

    ## crop image from the edge of the manhole is 1000mm
    size_box = max((xmax-xmin), (ymax-ymin)) + 2000
    x_crop_min_ori = xmin - 1000
    x_crop_max_ori = xmax + 1000
    y_crop_min_ori = ymin - 1000
    y_crop_max_ori = ymax + 1000
    
    if x_crop_min_ori < 0: 
      x_crop_min_ori = 0
      x_crop_max_ori = x_crop_min_ori + size_box

    if x_crop_max_ori > 4478: 
      x_crop_max_ori = 4478
      x_crop_min_ori = x_crop_max_ori - size_box

    if y_crop_min_ori < 0:
      y_crop_min_ori = 0
      y_crop_max_ori = y_crop_min_ori + size_box

    if y_crop_max_ori > 9999:
      y_crop_max_ori = 9999
      y_crop_min_ori = y_crop_max_ori - size_box

    crop_manhole_1000mm = img[int(y_crop_min_ori):int(y_crop_max_ori), int(x_crop_min_ori):int(x_crop_max_ori)]

    crop_manhole = img[int(max(0,ymin)):int(min(9999,ymax)), int(max(0,xmin)):int(min(4478,xmax))]

    ## calculate Radius, (x,y) center point
    R, x_center, y_center = (xmax+ymax-xmin-ymin)/4.0, (xmin - x_crop_min + xmax - x_crop_min)/2.0, (ymin - y_crop_min + ymax - y_crop_min)/2.0

    if xmin - 20 < 0:
        if xmax - xmin > (ymax - ymin)/2:
            R = (ymax - ymin)/2
            x_center, y_center = xmax - R, (ymax + ymin)/2.0
            x_center, y_center = x_center - x_crop_min, y_center - y_crop_min
        else:
            R, x_center, y_center = find_center_radius_circle(xmin, ymin, xmax, (ymin+ymax)/2, xmin, ymax)
            x_center, y_center = x_center - x_crop_min, y_center - y_crop_min
    elif xmax + 20 > 4478:
        if xmax - xmin > (ymax - ymin)/2:
            R = (ymax - ymin)/2
            x_center, y_center = xmin + R, (ymax + ymin)/2.0
            x_center, y_center = x_center - x_crop_min, y_center - y_crop_min
        else:
            R, x_center, y_center = find_center_radius_circle(xmin, (ymin+ymax)/2, xmax, ymax, xmax, ymin)
            x_center, y_center = x_center - x_crop_min, y_center - y_crop_min

    elif ymin - 20 < 0:
        if ymax - ymin > (xmax - xmin)/2:
            R = (xmax - xmin)/2
            x_center, y_center = (xmax + xmin)/2.0, ymax - R 
            x_center, y_center = x_center - x_crop_min, y_center - y_crop_min
        else:
            R, x_center, y_center = find_center_radius_circle(xmin, ymin, (xmax + xmin)/2, ymax, xmax, ymin)
            x_center, y_center = x_center - x_crop_min, y_center - y_crop_min
    elif ymax + 20 > 9999:
        if ymax - ymin > (xmax - xmin)/2:
            R = (xmax - xmin)/2
            x_center, y_center = (xmax + xmin)/2.0, ymin + R 
            x_center, y_center = x_center - x_crop_min, y_center - y_crop_min
        else:
            R, x_center, y_center = find_center_radius_circle(xmin, ymax, (xmax + xmin)/2, ymin, xmax, ymax)
            x_center, y_center = x_center - x_crop_min, y_center - y_crop_min

    R, x_center, y_center = int(R), int(x_center), int(y_center)
    
    pavement_area_250mm = int(250*(2*3.14*(R+125)))
    pavement_area_180mm = int(180*(2*3.14*(R+90)))

    img_full_white = np.zeros_like(crop_img_1) + 255
    img_circle_black = cv2.circle(img_full_white, (x_center, y_center), R+257, (0,0,0), -1)
    img_circle_pavament = np.array(crop_img_1, dtype=np.float32) + np.array(img_circle_black, dtype=np.float32)
    img_circle_pavament = np.where(img_circle_pavament > 255, 255, img_circle_pavament)
    crop_manhole_crack_250mm = cv2.circle(img_circle_pavament, (x_center, y_center), R+7, (255,255,255), -1)

    img_full_white = np.zeros_like(crop_img_1) + 255
    img_circle_black = cv2.circle(img_full_white, (x_center, y_center), R+187, (0,0,0), -1)
    img_circle_pavament = np.array(crop_img_1, dtype=np.float32) + np.array(img_circle_black, dtype=np.float32)
    img_circle_pavament = np.where(img_circle_pavament > 255, 255, img_circle_pavament)
    crop_manhole_crack_180mm = cv2.circle(img_circle_pavament, (x_center, y_center), R+7, (255,255,255), -1)

    return  crop_manhole, crop_manhole_250mm, crop_manhole_1000mm, crop_manhole_crack_180mm, crop_manhole_crack_250mm, R, x_center + x_crop_min, y_center + y_crop_min, pavement_area_250mm, pavement_area_180mm

def make_manhole_excel_file(file_path, path_original, submission_sum_list):
    workbook = xlsxwriter.Workbook(file_path)

    cell_format = workbook.add_format()
    cell_format.set_align('center')
    cell_format.set_align('vcenter')

    cell_format_check = workbook.add_format({'bold': True, 'font_color': 'red'})
    cell_format_check.set_align('center')
    cell_format_check.set_align('vcenter')

    worksheet = workbook.add_worksheet()
    worksheet.write_row(0, 0, ['Version', today.strftime("%m/%d/%Y")])
    worksheet.write_row(1, 0, ['경로', '{}'.format(path_original)])
    worksheet.write_row(2, 0, ['결과사이즈_width_height_mm', '3000', '3000'])
    worksheet.write_row(3, 0, ['결과해상도_width_height_mm', '1', '5'])
    worksheet.write_row(4, 0, ['필터사이즈_mm', '20']),
    worksheet.write_row(5, 0, ['ID_Iris_Manhole','ID_manhole', 'Severity_level', '조사연장[m]', 'x[m]', 'y[m]', 'width[m]', 'height[m]', 'R[m]', 'Crack_percent[%]', 'Manhole_img'])
    # worksheet.set_default_row(100)
    worksheet.set_column('K:K', 60)
    worksheet.set_column('A:A', 30)
    worksheet.set_column('B:B', 20)


    for i, row in enumerate(submission_sum_list):
        worksheet.set_row(i+6, 250)
        if float(row[-1][0:-1]) < 95: worksheet.write_row(i+6, 0 , row[0:-2], cell_format_check)
        else: worksheet.write_row(i+6, 0 , row[0:-2], cell_format)
        worksheet.insert_image('K{}'.format(i+7), row[-2], {'x_scale': 0.22, 'y_scale': 0.22})

    workbook.close()

def main(args=None):
    # load folder directory
    load_folder_path = get_main_source_dir(root_dir=None, name = "Open image Directory")
    save_path = get_main_source_dir(root_dir=None, name = "Open result_manhole Directory")

    if manhole_detection == True:

        save_folder_path = os.path.join(save_path, "{}_manhole_results".format(os.path.basename(load_folder_path).split("_")[0]))
        save_excel_sum_folder_path = os.path.join(save_folder_path, "{}_profile_excel".format(os.path.basename(load_folder_path).split("_")[0]))
        save_csv_sum_folder_path = os.path.join(save_folder_path, "{}_extractor_excel".format(os.path.basename(load_folder_path).split("_")[0]))
        if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
        if not os.path.exists(save_excel_sum_folder_path):
                    os.makedirs(save_excel_sum_folder_path)
        if not os.path.exists(save_csv_sum_folder_path):
                    os.makedirs(save_csv_sum_folder_path)
        count_image = 0
        count_manhole =0

        for path_original, between, fnames in os.walk(load_folder_path):
            elem_to_find = os.path.basename(path_original) + '_표면결함'
            res1 = any(elem_to_find in sublist for sublist in between)
            if res1:
                load_img_folder_path = os.path.join(path_original, elem_to_find)
                print(load_img_folder_path)
                # making new folder for save as the results

                save_folder_path_i = os.path.join(save_folder_path, load_img_folder_path.split("/")[-2])
                # if not os.path.exists(save_folder_path_i):
                #     os.makedirs(save_folder_path_i)
                list_check_manually = []
                submission_3D_sum = []
                submission_sum_list = []
                for path, between, fnames in os.walk(load_img_folder_path):
                    # print(path, between, fnames)
                    for fname in fnames:
                        if fname.endswith('.jpg'):
                            save_imgs_folder_path, save_annotations_folder_path, save_excel_folder_path, save_imgs_original_folder_path, \
                            save_crop_manhole_folder_path, save_imgs_check_folder_path, save_crack_manhole_folder_path, \
                            save_crop_manhole_250mm_folder_path, save_crop_manhole_1000mm_folder_path = make_folder(save_folder_path_i)
                            fnames.sort()
                            # print(fnames)
                            for fname in fnames:
                                if fname.endswith('.jpg'):
                                    start_time = time.time()
                                    count_image += 1
                                    img_input_path = os.path.join(path, fname)
                                    fname_base = fname.split(".")[0]
                                    print(img_input_path)

                                    class_names_manhole_list = []
                                    bboxes_manhole_list = []

                                    x, orig_img = load_test(img_input_path, real_width = 4479, real_height = 10000)
                                    x = x.as_in_context(ctx)
                                    # print(x.shape)
                                    box_ids, scores, bboxes = net_4479_10000(x)
                                    # print(box_ids[0], scores[0], bboxes[0])
                                    result_manhole_4479_10000 = get_bbox_4479_10000(orig_img, x, bboxes[0], scores[0], box_ids[0], class_names=net_4479_10000.classes, thresh=0.7)
                                    # class_names, scores, bboxes = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net_4479_10000.classes, thresh=0.5, absolute_coordinates=False)
                                    # print(len(result_manhole_4479_10000))
                                    if len(result_manhole_4479_10000) > 1:
                                        result_manhole_4479_10000 = check_overlap_prediction(result_manhole_4479_10000)
                                    # print(len(result_manhole_4479_10000))

                                    img_manhole = orig_img.copy()
                                    if result_manhole_4479_10000 == []: img_manhole = img_manhole.asnumpy()
                                    for i, result_manhole_4479_10000_i in enumerate(result_manhole_4479_10000):
                                        [_,_, bboxes] = result_manhole_4479_10000_i
                                        # manhole = orig_img[int(bboxes[i][1]):int(bboxes[i][3]), int(bboxes[i][0]):int(bboxes[i][2])]
                                        x_center = int((int(bboxes[0])+int(bboxes[2]))/2.0)
                                        y_center = int((int(bboxes[1])+int(bboxes[3]))/2.0)

                                        x_crop_min = x_center - 800
                                        x_crop_max = x_center + 800
                                        y_crop_min = y_center - 800
                                        y_crop_max = y_center + 800

                                        if x_crop_min < 0:
                                            x_crop_min = 0
                                            x_crop_max = 1600

                                        if x_crop_max > 4478:
                                            x_crop_min = 4478 - 1600
                                            x_crop_max = 4478

                                        if y_crop_min < 0:
                                            y_crop_min = 0
                                            y_crop_max = 1600

                                        if y_crop_max > 9999:
                                            y_crop_min = 9999 - 1600
                                            y_crop_max = 9999

                                        manhole_crop = orig_img[int(y_crop_min):int(y_crop_max), int(x_crop_min):int(x_crop_max)]
                                        # print("--------------------------------------")
                                        manhole, manhole_original = transform_test(manhole_crop, real_width = 1600, real_height = 1600)
                                        # print(manhole.shape)

                                        ## detecting manhole 1600_1600
                                        box_ids_manhole, scores_manhole, bboxes_manhole = net_1600_1600(manhole.as_in_context(ctx))
                                        result_manhole_1600_1600 = get_bbox_1600_1600(img_manhole, manhole, bboxes_manhole[0], scores_manhole[0],
                                                                                                            box_ids_manhole[0], class_names=net_1600_1600.classes, thresh=0.5, absolute_coordinates=True,
                                                                                                            x_crop_min = x_crop_min, y_crop_min = y_crop_min)

                                        # print(len(result_manhole_1600_1600))
                                        if len(result_manhole_1600_1600) > 1:
                                            result_manhole_1600_1600 = check_overlap_prediction(result_manhole_1600_1600)
                                            if len(result_manhole_1600_1600) > 1:
                                                result_manhole_1600_1600 = check_overlap_prediction_1600_1600(result_manhole_1600_1600, result_manhole_4479_10000_i)
                                        # print(len(result_manhole_1600_1600))

                                        ## Classification manhole 1600_1600
                                        box_ids_manhole_classification, scores_manhole_classification, bboxes_manhole_classification = net_1600_1600_detail(manhole.as_in_context(ctx))
                                        result_manhole_1600_1600_classification = get_bbox_1600_1600(img_manhole, manhole, bboxes_manhole_classification[0], scores_manhole_classification[0],
                                                                                                            box_ids_manhole_classification[0], class_names=net_1600_1600_detail.classes, thresh=0.6, absolute_coordinates=True,
                                                                                                            x_crop_min = x_crop_min, y_crop_min = y_crop_min)

                                        # print(len(result_manhole_1600_1600_classification))
                                        if len(result_manhole_1600_1600_classification) > 1:
                                            result_manhole_1600_1600_classification = check_overlap_prediction_classification(result_manhole_1600_1600_classification)
                                            if len(result_manhole_1600_1600_classification) > 1:
                                                result_manhole_1600_1600_classification = check_overlap_prediction_1600_1600(result_manhole_1600_1600_classification, result_manhole_4479_10000_i)
                                        # print(len(result_manhole_1600_1600_classification))


                                        if len(result_manhole_1600_1600) == 1 and len(result_manhole_1600_1600_classification) == 1:
                                            basic_name_4479_10000, score_4479_10000 =  (result_manhole_4479_10000_i[0].split('_'))[2], result_manhole_4479_10000_i[1]
                                            basic_name_1600_1600_classification, score_1600_1600_classification = (result_manhole_1600_1600_classification[0][0].split('_'))[2], result_manhole_1600_1600_classification[0][1]
                                            # print(basic_name_4479_10000, basic_name_1600_1600_classification)

                                            if basic_name_4479_10000 == basic_name_1600_1600_classification and min(float(score_4479_10000[:-1]), float(score_1600_1600_classification[:-1])) > 90:
                                                result_manhole_1600_1600_classification[0][2] = result_manhole_1600_1600[0][2]
                                            else:
                                                result_manhole_1600_1600_classification[0][2] = result_manhole_1600_1600[0][2]
                                                print(" need to check manually again", fname)
                                                img_check_output_path = os.path.join(save_imgs_check_folder_path, fname)
                                                cv2.imwrite(img_check_output_path, orig_img.asnumpy() if isinstance(orig_img, mx.nd.NDArray) else orig_img)
                                                list_check_manually.append(fname)
                                                # continue
                                        else:
                                            print(" need to check manually again", fname)
                                            img_check_output_path = os.path.join(save_imgs_check_folder_path, fname)
                                            cv2.imwrite(img_check_output_path, orig_img.asnumpy() if isinstance(orig_img, mx.nd.NDArray) else orig_img)
                                            list_check_manually.append(fname)
                                            # continue

                                        for j, result_manhole_1600_1600_i in enumerate(result_manhole_1600_1600_classification):

                                            xmin, ymin, xmax, ymax = [int(x) for x in result_manhole_1600_1600_i[2]]

                                            # detecting manhole crack 1600_1600
                                            manhole_crop_original, manhole_crop_250mm, manhole_crop_1000mm,\
                                            crop_manhole_crack_180mm, manhole_crop_crack, \
                                            R, x_center, y_center, \
                                            pavement_area, pavement_area_180mm = crop_manhole_for_calculate_crack(orig_img, xmin, ymin, xmax, ymax)

                                            ## excepting small manholes R < 250mm
                                            if R < 250: continue
                                            class_names_manhole_list.append(result_manhole_1600_1600_i[0])
                                            bboxes_manhole_list.append(result_manhole_1600_1600_i[2])
                                            count_manhole += 1
                                            ## detecting manhole crack 250mm
                                            crack_percent = detect_manhole_crack(manhole_crop_crack, save_crack_manhole_folder_path, fname, pavement_area, x_center, y_center, result_manhole_1600_1600_i[0], write_image = save_image_manhole_crack_2500mm)
                                            crack_percent_180mm = detect_manhole_crack(crop_manhole_crack_180mm, save_crack_manhole_folder_path, '180mm_'+fname, pavement_area_180mm, x_center, y_center, result_manhole_1600_1600_i[0], write_image = save_image_manhole_crack_1800mm)
                                            crack_percent_mix = (crack_percent_180mm*2 + crack_percent)/3.0

                                            ## change xy
                                            x_center_change_xy = int(x_center) -218
                                            y_center_change_xy = 10000 - int(y_center) + 113
                                            # x_center_change_xy = int(x_center*realwidth/4479)
                                            # y_center_change_xy = 10000 - int(y_center)

                                            ## write images

                                            if save_image_result == True:
                                                img_manhole = draw_result_image(img_manhole, result_manhole_1600_1600_i)

                                            if save_image_manhole_crop == True:
                                                img_output_path = os.path.join(save_crop_manhole_folder_path, fname.split('.')[0] + '_{}_{}_{}.jpg'.format(change_name(result_manhole_1600_1600_i[0]), x_center_change_xy, y_center_change_xy))
                                                cv2.imwrite(img_output_path, manhole_crop_original)

                                            if save_image_manhole_crop_2500mm == True:
                                                img_output_path_250mm = os.path.join(save_crop_manhole_250mm_folder_path, fname.split('.')[0] + '_{}_{}_{}.jpg'.format(change_name(result_manhole_1600_1600_i[0]), x_center_change_xy, y_center_change_xy))
                                                cv2.imwrite(img_output_path_250mm, manhole_crop_250mm)

                                            if save_image_manhole_crop_10000mm == True:
                                                img_output_path_1000mm = os.path.join(save_crop_manhole_1000mm_folder_path, fname.split('.')[0] + '_{}_{}_{}.jpg'.format(change_name(result_manhole_1600_1600_i[0]), x_center_change_xy, y_center_change_xy))
                                                cv2.imwrite(img_output_path_1000mm, manhole_crop_1000mm)

                                            ## writing excel list
                                            submission_3D_sum.append("{}, {}, {}".format(int((fname.split('.')[0]).split('_')[-1][1:-3]) - 10 ,
                                                                                        round(x_center_change_xy*0.001, 4),
                                                                                        round(y_center_change_xy*0.001, 4)))

                                            id_iris_manhole =  os.path.basename(path_original).replace('_', '-') + '-{0:03}'.format(count_manhole)
                                            submission_sum_list.append([id_iris_manhole,
                                                                        change_name(result_manhole_1600_1600_i[0]),
                                                                        crack_serverity(crack_percent_mix),
                                                                        int((fname.split('.')[0]).split('_')[-1][1:-3]) - 10 ,
                                                                        round(x_center_change_xy*0.001, 4),
                                                                        round(y_center_change_xy*0.001, 4),
                                                                        round(int(xmax - xmin)*0.001, 4),
                                                                        round(int(ymax - ymin)*0.001, 4),
                                                                        round(int(R)*0.001, 4),
                                                                        round(crack_percent_mix, 2),
                                                                        img_output_path_250mm,
                                                                        result_manhole_1600_1600_i[1]])

                                    img_output_path = os.path.join(save_imgs_folder_path, fname)
                                    img_original_output_path = os.path.join(save_imgs_original_folder_path, fname)

                                    if save_image_result == True:
                                        cv2.imwrite(img_output_path, img_manhole.asnumpy() if isinstance(img_manhole, mx.nd.NDArray) else img_manhole)
                                    if save_image_original_4479_10000 == True:
                                        cv2.imwrite(img_original_output_path, orig_img.asnumpy() if isinstance(orig_img, mx.nd.NDArray) else orig_img)


                                    path_annot = os.path.join(save_annotations_folder_path, "{}.xml".format(fname_base))
                                    GenerateXML(path = path_annot, filename = fname, width = 4479, height = 10000,
                                                    name_manhole_list = class_names_manhole_list, bboxes_list = bboxes_manhole_list)

                                    print("--- %s seconds per an image---" % (time.time() - start_time))
                                    print("******************************************************************************************")
                            break

                file_path = os.path.join(save_excel_folder_path, 'manhole_results_{}.xlsx'.format(os.path.basename(path_original)))
                file_total_path = os.path.join(save_excel_sum_folder_path, 'manhole_results_{}.xlsx'.format(os.path.basename(path_original)))
                make_manhole_excel_file(file_path, path_original, submission_sum_list)
                make_manhole_excel_file(file_total_path, path_original, submission_sum_list)

                ## making 3D csv file
                submission_3D_slope_0_sum = ("Version, {} \n".format(today.strftime("%m/%d/%Y")) +
                    "경로, {} \n".format(path_original) +
                    "결과사이즈_width_height_mm, 3000, 3000 \n" +
                    "결과해상도_width_height_mm, 1, 5 \n" +
                    "필터사이즈_mm, 20 \n" +
                    "Remove slope, 0 \n" +
                    "조사연장[m], x[m], y[m]\n" +
                    "\n".join(submission_3D_sum))
                file_path = os.path.join(save_excel_folder_path, 'manhole_results_3D_{}_0.csv'.format(os.path.basename(path_original)))
                file_total_path = os.path.join(save_csv_sum_folder_path, 'manhole_results_3D_{}_0.csv'.format(os.path.basename(path_original)))
                with open(file_path, "w") as f:
                    f.write(submission_3D_slope_0_sum)
                with open(file_total_path, "w") as f:
                    f.write(submission_3D_slope_0_sum)

                ## making 3D csv file
                submission_3D_slope_1_sum = ("Version, {} \n".format(today.strftime("%m/%d/%Y")) +
                                "경로, {} \n".format(path_original) +
                                "결과사이즈_width_height_mm, 3000, 3000 \n" +
                                "결과해상도_width_height_mm, 1, 5 \n" +
                                "필터사이즈_mm, 20 \n" +
                                "Remove slope, 1 \n" +
                                "조사연장[m], x[m], y[m]\n" +
                                "\n".join(submission_3D_sum))
                file_path = os.path.join(save_excel_folder_path, 'manhole_results_3D_{}_1.csv'.format(os.path.basename(path_original)))
                file_total_path = os.path.join(save_csv_sum_folder_path, 'manhole_results_3D_{}_1.csv'.format(os.path.basename(path_original)))
                with open(file_path, "w") as f:
                    f.write(submission_3D_slope_1_sum)
                with open(file_total_path, "w") as f:
                    f.write(submission_3D_slope_1_sum)
                print("Saved manhole_results_3D.csv to ", save_excel_sum_folder_path)

                # print("-----list_check_manually--------")
                # for i in list_check_manually: print(i)
        print('+++++++++++++{} images are analysed++++++++++++++++++++'.format(count_image))
        print('+++++++++++++{} manholes are analysed++++++++++++++++++++'.format(count_manhole))

    Calculate_GPS_IRIS_4479realwidth_newGPS.calculate_GPS(save_path)
    combine_GPS_manholes.combine_GPS_manholes(save_path)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    start_time = time.time()
    main()
    print('+++++++++++++{} time for analysing++++++++++++++++++++'.format(round(time.time()-start_time, 0)))
