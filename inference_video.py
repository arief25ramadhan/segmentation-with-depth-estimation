import cv2
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import MobileNetv2, MTLWRefineNet
import torch.nn as nn
import matplotlib.colors as co
import matplotlib.cm as cm

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

def inference(img):
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

    return segm, depth


def depth_to_rgb(depth):
    normalizer = co.Normalize(vmin=0, vmax=80)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im

result_video = []
video_capture = cv2.VideoCapture('media/kitti_07.mp4')

if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

while (video_capture.isOpened()):

    ret, frame = video_capture.read()

    if not ret:
        break

    # Perform inference on frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = np.array(frame)
    h, w, _ = image.shape
    seg, depth = inference(image)
    result_video.append(cv2.cvtColor(cv2.vconcat([image, seg, depth_to_rgb(depth)]), cv2.COLOR_BGR2RGB))

out = cv2.VideoWriter('media/kitti_07_output.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (w,3*h))

for i in range(len(result_video)):
    out.write(result_video[i])
out.release()