import base64
import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from typing_extensions import Annotated

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

curr_dir = os.path.dirname(__file__)
checkpoint = curr_dir + "/model/sam_vit_h.pth"
model_type = "vit_h"

# Download the model if it doesn't exist
if not os.path.exists(checkpoint):
    import requests
    from tqdm import tqdm

    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(checkpoint, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


def set_device(dev: str):
    global device
    if dev:
        device = dev
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"


@app.post("/api/embedding")
async def create_upload_file(file: Annotated[bytes, File()]):
    # Read the image file
    image_data = Image.open(io.BytesIO(file))
    nparr = np.array(image_data)

    # Embedding generation
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    _ = sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(nparr)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    print(image_embedding.shape)

    # 将float32转换为float8 (1位符号，4位指数，3位小数)
    def float32_to_float8(arr):
        # 将数组展平
        flat_arr = arr.ravel()
        
        # 创建输出数组
        uint8_arr = np.zeros_like(flat_arr, dtype=np.uint8)
        
        # 处理特殊情况（使用向量化操作）
        is_nan = np.isnan(flat_arr)
        is_zero = flat_arr == 0
        is_pos_inf = np.isposinf(flat_arr)
        is_neg_inf = np.isneginf(flat_arr)
        
        uint8_arr[is_nan] = 0xFF
        uint8_arr[is_zero] = 0x00
        uint8_arr[is_pos_inf] = 0x7F
        uint8_arr[is_neg_inf] = 0xFF
        
        # 处理正常值
        normal_mask = ~(is_nan | is_zero | is_pos_inf | is_neg_inf)
        normal_vals = flat_arr[normal_mask]
        
        if len(normal_vals) > 0:
            # 计算符号位
            signs = np.signbit(normal_vals).astype(np.uint8) << 7
            
            # 计算指数
            abs_vals = np.abs(normal_vals)
            exps = np.floor(np.log2(abs_vals)).astype(np.int32)
            exps = np.clip(exps + 7, 0, 15) << 3
            
            # 计算尾数
            fracs = ((abs_vals / (2 ** (exps >> 3 - 7))) - 1.0) * 8
            fracs = fracs.astype(np.uint8) & 0x7
            
            # 组合所有位
            uint8_arr[normal_mask] = signs | exps | fracs
        
        return uint8_arr.reshape(arr.shape)

    reduced_precision = float32_to_float8(image_embedding)
    embedding_base64 = base64.b64encode(reduced_precision.tobytes()).decode("utf-8")

    return [embedding_base64]


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Server for embedding generation using SAM."
    )

    parser.add_argument("--device", type=str, help="The device to run the model on.")

    args = parser.parse_args()
    set_device(args.device)

    uvicorn.run(app, host="0.0.0.0", port=3000)
