from argparse import Namespace

config = Namespace()

# path to where resnet101 pretrained model is stored
config.RESNET_PRETRAINED_DIR = "/pretrained_models/resnet101-cd907fc2.pth"

# RESNET101 with FPN
config.IN_CHANNELS = (256, 512, 1024, 2048)
config.OUT_CHANNELS = 256
config.STAGES = (1, 2, 3, 4)

# FasterRCNNcddod
config.IMAGE_MEAN=(0.485, 0.456, 0.406)
config.IMAGE_STD =(0.229, 0.224, 0.225)

# LOSSES
#fpa loss
config.FPA_WEIGHT=True
config.FPA_MOD_WEIGHTS=(0.0, 0.0, 0.0, 1)

# focal loss
config.GAMMA = 5.0
config.ALPHA = 0.5
# local classifier loss if using swda
config.MSE = True
config.WEIGHT = False

# FASTERRCNN
# 1. rpn
config.MIN_SIZE = 600
config.MAX_SIZE = 1200
config.RPN_PRE_NMS_TOP_N_TRAIN = 2000
config.RPN_PRE_NMS_TOP_N_TEST = 1000
config.RPN_POST_NMS_TOP_N_TRAIN = 2000
config.RPN_POST_NMS_TOP_N_TEST = 1000
config.RPN_NMS_THRESH=0.7

# gradient reversal layer
config.LAMBDA = 1.0
