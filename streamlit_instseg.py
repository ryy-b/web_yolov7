import sys
from glob import glob
sys.path.append("/workspaces/web_yolov7/yolov7/")
for p in glob("/workspaces/web_yolov7/yolov7/*"):
    sys.path.append(p)

import io
import cv2
import yaml
import torch
import random
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from torchvision import transforms
from memory_profiler import profile
import warnings;warnings.simplefilter('ignore')

from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression_mask_conf
from yolov7.detectron2.structures import Boxes
from yolov7.detectron2.modeling.poolers import ROIPooler
from yolov7.detectron2.layers import paste_masks_in_image
from yolov7.detectron2.utils.memory import retry_if_cuda_oom



# 画像をモデルに入れて、推論させる
def inference(image_path, model):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('./yolov7/data/hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

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
    model.eval()
    output = model(image)

    return output, image, model, hyp



def show_result(output, image, model, hyp):

    inf_out, attn, bases, sem_output = output['test'], output['attn'], output['bases'], output['sem']

    # bases : 物体検出用にエンコードされた、特徴マップ
    # sem_output : セグメンテーション用にエンコードされた特徴マップ
    bases = torch.cat([bases, sem_output], dim=1)
    
    height = image.shape[2]
    width = image.shape[3]
    names = model.names
    pooler_scale = model.pooler_scale

    # Region of Interest (関心領域) 物体が写っているであろう領域を切り出す
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)

    # 信頼度の高い結果を抽出
    output, output_mask, _, _, _ = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)
    pred, pred_masks = output[0], output_mask[0]
    bboxes = Boxes(pred[:, :4])

    # 予測されたマスクを hyp['mask_resolution']×hyp['mask_resolution']のサイズにreshape
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])

    # 画像にマスクを貼り付ける
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()

    # pred(tensor) : 5つの列から構成
    # 0から3列目が座標、4列目が存在確率、5列目が物体のクラス
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()

    # image.shape : (batch_size, channels, height, width)
    # permute(1, 2, 0)で、(height, width, channels)に変換
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR) # OpenCVの仕様上、RGBではなくBGRにする必要がある
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
    pnimg = nimg.copy()

    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if conf < 0.25: continue # 信頼度が低いとき、表示しない
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5 # 透明感を出して、見やすくする
        pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cls_name = names[int(cls)]
        text = f"{cls_name} ({conf:.2f})"

        # フォントの種類とサイズ
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7

        # アンチエイリアシングを有効にする
        thickness = 1
        lineType = cv2.LINE_AA

        pnimg = cv2.putText(pnimg, text, (bbox[0], bbox[1]), fontFace, fontScale, color, thickness, lineType)

    pnimg = cv2.resize(pnimg, (pnimg.shape[1] * 4, pnimg.shape[0] * 4))

    return pnimg



# class nameをグリッド状に表示
def display_class_names_in_grid(class_names):

    st.markdown(f"**:blue[Object List that YOLOv7 can recognize ({len(class_names)} classes)]**")

    list_items = ""
    for c in class_names:
        list_items += f"<div>{c}</div>"

    # グリッド上にして、背景をグラデーションにした
    st.markdown(f"""
    <style>
    .wrapper {{
        display: grid;
        grid-template-columns: repeat(12, 120px);
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 10px;
    }}
    .wrapper div {{
        border: 1px solid black;
        padding: 10px;
        background-color: white;
        text-align: center;
    }}
    </style>

    <div class="wrapper">
        {list_items}
    </div>
    """, unsafe_allow_html=True)



# ページのレイアウトを調整
st.set_page_config(layout="wide")

st.title('YOLOv7 Instance Segmentation')

# 画像ファイルのuploaderを表示
uploaded_files = st.file_uploader('Choose some images!', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# デバイス設定とモデル読み込み
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights = torch.load('./yolov7/yolov7-mask.pt')
model = weights['model'].to(device).float().eval()
names = model.names

# 識別可能なクラスをグリッド状に表示
display_class_names_in_grid(names)

# # 2つの列にページを分割
left_column, right_column = st.columns(2)

if uploaded_files is not None:
    input_images = []
    results = []
    for i, uploaded_file in enumerate(uploaded_files):

        output, image, model, hyp = inference(uploaded_file, model)
        result_image = show_result(output, image, model, hyp)
        results.append(result_image)
        image = Image.open(uploaded_file)
        input_images.append(image)

    # 結果を表示
    for result in results:
        right_column.image(result, caption='Result Image (MS COCO dataset)', use_column_width=True)
        
    # 入力画像を表示
    for image in input_images:
        left_column.image(image, caption='Original Image (MS COCO dataset)', use_column_width=True)