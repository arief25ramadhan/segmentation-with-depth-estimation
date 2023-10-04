import cv2
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import MobileNetv2, MTLWRefineNet
import torch.nn as nn

img_path = 'dataset/nyud/rgb/000001.png'
img = np.array(Image.open(img_path))
gt_segm = np.array(Image.open('dataset/nyud/masks/000001.png'))
gt_depth = np.array(Image.open('dataset/nyud/depth/000001.png'))

def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

# Pre-processing and post-processing constants #
CMAP = np.load('dataset/cmap_nyud.npy')
DEPTH_COEFF = 5000. # to convert into metres
HAS_CUDA = torch.cuda.is_available()
IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
MAX_DEPTH = 8.
MIN_DEPTH = 0.
NUM_CLASSES = 40
NUM_TASKS = 2 # segm + depth

# Load Model
encoder = MobileNetv2()
num_classes = (40, 1)
decoder = MTLWRefineNet(encoder._out_c, num_classes)

hydranet = nn.DataParallel(nn.Sequential(encoder, decoder).cuda()) # Use .cpu() if you prefer a slow death
model_path = "checkpoint.pth.tar"
checkpoint = torch.load(model_path)
hydranet.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():
    img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
    if HAS_CUDA:
        img_var = img_var.cuda()
    segm, depth = hydranet(img_var)
    segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                      img.shape[:2][::-1],
                      interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                       img.shape[:2][::-1],
                       interpolation=cv2.INTER_CUBIC)
    segm = CMAP[segm.argmax(axis=2) + 1].astype(np.uint8)
    depth = np.abs(depth)
    
# depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)
colormap = plt.get_cmap('plasma')
heatmap = (colormap(depth) * 2**16).astype(np.uint16)[:,:,:3]
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

cv2.imwrite('depth_save.png',heatmap)
cv2.imwrite('segm_save.png',segm)