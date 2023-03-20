from glob import glob
import sys
sys.path.append("/workspaces/Web_InstSeg/yolov7/")
for p in glob("/workspaces/Web_InstSeg/yolov7/*"):
    sys.path.append(p)

import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np

from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression_mask_conf

from yolov7.detectron2.modeling.poolers import ROIPooler
from yolov7.detectron2.structures import Boxes
from yolov7.detectron2.utils.memory import retry_if_cuda_oom
from yolov7.detectron2.layers import paste_masks_in_image
import warnings;warnings.simplefilter('ignore')
import streamlit as st
from memory_profiler import profile
from PIL import Image
import io
import tempfile


def inference(image_path):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('./yolov7/data/hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
        
    weights = torch.load('./yolov7/yolov7-mask.pt')
    model = weights['model'].to(device).float().eval()

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(uploaded_file.read())
        image_path = f.name

    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image = letterbox(image, 640, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image).float()
    image = image.unsqueeze(0).to(device)

    output = model(image)

    return output, image, model, hyp



def show_result(output, image, model, hyp):

    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']
    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = image.shape
    names = model.names
    pooler_scale = model.pooler_scale
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)

    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)
    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    bboxes = Boxes(pred[:, :4])
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
    pnimg = nimg.copy()

    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if conf < 0.25:
            continue
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cls_name = names[int(cls)]
        text = f"{cls_name} ({conf:.2f})"

        # フォントの種類とサイズ
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5

        # アンチエイリアシングを有効にする
        thickness = 1
        lineType = cv2.LINE_AA

        pnimg = cv2.putText(pnimg, text, (bbox[0], bbox[1]), fontFace, fontScale, color, thickness, lineType)
    
    pnimg = cv2.resize(pnimg, (pnimg.shape[1] * 2, pnimg.shape[0] * 2))

    return pnimg


# ページのレイアウトを調整
st.set_page_config(layout="wide")
# Streamlitアプリケーションを定義する
st.title('YOLOv7 Instance Segmentation')
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    output, image, model, hyp = inference(uploaded_file)
    
    names = model.names

    st.write("<h1>'Objects List that Yolov7 can recognize is showed below.'</h1>", unsafe_allow_html=True)
    st.write(names, unsafe_allow_html=True)

    result_image = show_result(output, image, model, hyp)
    st.image(result_image, channels='RGB')