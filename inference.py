import cv2
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_path = '/content/nyud/rgb/000001.png'
img = np.array(Image.open(img_path))
gt_segm = np.array(Image.open('/content/nyud/masks/000001.png'))
gt_depth = np.array(Image.open('/content/nyud/depth/000001.png'))

def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

# Pre-processing and post-processing constants #
CMAP = np.load('cmap_nyud.npy')
DEPTH_COEFF = 5000. # to convert into metres
HAS_CUDA = torch.cuda.is_available()
IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
MAX_DEPTH = 8.
MIN_DEPTH = 0.
NUM_CLASSES = 40
NUM_TASKS = 2 # segm + depth

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
plt.figure(figsize=(18, 12))
plt.subplot(151)
plt.imshow(img)
plt.title('orig img')
plt.axis('off')
plt.subplot(152)
plt.imshow(CMAP[gt_segm + 1])
plt.title('gt segm')
plt.axis('off')
plt.subplot(153)
plt.imshow(segm)
plt.title('pred segm')
plt.axis('off')
plt.subplot(154)
plt.imshow(gt_depth / DEPTH_COEFF, cmap='plasma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
plt.title('gt depth')
plt.axis('off')
plt.subplot(155)
plt.imshow(depth, cmap='plasma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
plt.title('pred depth')
plt.axis('off');
