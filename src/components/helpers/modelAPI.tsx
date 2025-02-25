// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { Tensor } from "onnxruntime-web";
import {
  setParmsandQueryModelProps,
  queryModelReturnTensorsProps,
  modeDataProps,
  modelInputProps,
} from "./Interfaces";

const API_ENDPOINT = process.env.API_ENDPOINT;

const setParmsandQueryModel = ({
  width,
  height,
  uploadScale,
  imgData,
  handleSegModelResults,
  imgName,
}: setParmsandQueryModelProps) => {
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * uploadScale);
  canvas.height = Math.round(height * uploadScale);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(imgData, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(
    (blob) => {
      blob &&
        queryModelReturnTensors({
          blob,
          handleSegModelResults,
          imgName,
        });
    },
    "image/jpeg",
    1.0,
  );
};

const queryModelReturnTensors = async ({
  blob,
  handleSegModelResults,
  imgName,
}: queryModelReturnTensorsProps) => {
  if (!API_ENDPOINT) return;
  const req_data = new FormData();
  req_data.append("file", blob, imgName);

  const segRequest = fetch(`${API_ENDPOINT}/embedding`, {
    method: "POST",
    body: req_data,
  });

  segRequest.then(async (segResponse) => {
    const segJSON = await segResponse.json();
    const embedArr = segJSON.map((arrStr: string) => {
      const binaryString = window.atob(arrStr);
      const uint8arr = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        uint8arr[i] = binaryString.charCodeAt(i);
      }

      // 将uint8数组转换为float32数组
      const float32Arr = new Float32Array(uint8arr.length);
      for (let i = 0; i < uint8arr.length; i++) {
        const byte = uint8arr[i];

        // 解析float8的各个部分
        const sign = (byte & 0x80) ? -1 : 1;
        const exponent = (byte & 0x78) >> 3;
        const fraction = byte & 0x07;

        // 特殊值处理
        if (byte === 0x00) {
          float32Arr[i] = 0;  // Zero
          continue;
        }
        if (byte === 0xFF) {
          float32Arr[i] = NaN;  // NaN
          continue;
        }
        if (byte === 0x7F) {
          float32Arr[i] = Infinity;  // +Inf
          continue;
        }

        // 常规值转换
        const value = sign * Math.pow(2, exponent - 7) * (1 + fraction / 8);
        float32Arr[i] = value;
      }
      return float32Arr;
    });
    const lowResTensor = new Tensor("float32", embedArr[0], [1, 256, 64, 64]);
    handleSegModelResults({
      tensor: lowResTensor,
    });
  });
};

const getPointsFromBox = (box: modelInputProps) => {
  if (box.width === null || box.height === null) return;
  const upperLeft = { x: box.x, y: box.y };
  const bottomRight = { x: box.width, y: box.height };
  return { upperLeft, bottomRight };
};

const isFirstClick = (clicks: Array<modelInputProps>) => {
  return (
    (clicks.length === 1 && clicks[0].clickType === 1) ||
    (clicks.length === 2 && clicks.every((c) => c.clickType === 2))
  );
};

const modelData = ({
  clicks,
  tensor,
  modelScale,
  last_pred_mask,
}: modeDataProps) => {
  // 后续可以提供给用户调整参数

  const MAX_CLICK_RANGE = 500; // 设置最大点击范围为50像素
  const BOUNDARY_POINTS = 1; // 在周围添加8个负点击来限制范围

  const imageEmbedding = tensor;
  let pointCoords;
  let pointLabels;
  let pointCoordsTensor;
  let pointLabelsTensor;

  // Check there are input click prompts
  if (clicks) {
    let n = clicks.length;
    const clicksFromBox = clicks[0].clickType === 2 ? 2 : 0;

    // 为每个正点击创建周围的负点击
    const boundaryClicks = clicks.reduce((acc, click) => {
      if (click.clickType === 1) {  // 只处理普通点击，不处理框选
        const radius = MAX_CLICK_RANGE;
        // 在周围添加8个负点击
        for (let i = 0; i < BOUNDARY_POINTS; i++) {
          const angle = (i * 2 * Math.PI) / BOUNDARY_POINTS;
          const boundaryX = click.x + radius * Math.cos(angle);
          const boundaryY = click.y + radius * Math.sin(angle);

          // 确保边界点在图像范围内
          const x = Math.min(Math.max(0, boundaryX), modelScale.width);
          const y = Math.min(Math.max(0, boundaryY), modelScale.height);

          acc.push({
            x,
            y,
            width: null,
            height: null,
            clickType: 0  // 0 表示负点击
          });
        }
      }
      acc.push(click);  // 保留原始点击
      return acc;
    }, [] as modelInputProps[]);

    // 更新点击数量
    n = boundaryClicks.length;

    // If there is no box input, a single padding point with
    // label -1 and coordinates (0.0, 0.0) should be concatenated
    // so initialize the array to support (n + 1) points.
    pointCoords = new Float32Array(2 * (n + 1));
    pointLabels = new Float32Array(n + 1);

    // Check if there is a box input
    if (clicksFromBox) {
      pointCoords = new Float32Array(2 * (n + clicksFromBox));
      pointLabels = new Float32Array(n + clicksFromBox);
      const {
        upperLeft,
        bottomRight,
      }: {
        upperLeft: { x: number; y: number };
        bottomRight: { x: number; y: number };
      } = getPointsFromBox(boundaryClicks[0])!;
      pointCoords[0] = upperLeft.x * modelScale.samScale;
      pointCoords[1] = upperLeft.y * modelScale.samScale;
      pointLabels[0] = 2.0; // UPPER_LEFT
      pointCoords[2] = bottomRight.x * modelScale.samScale;
      pointCoords[3] = bottomRight.y * modelScale.samScale;
      pointLabels[1] = 3.0; // BOTTOM_RIGHT

      last_pred_mask = null;
    }

    // Add all clicks (including boundary clicks) and scale to what SAM expects
    for (let i = 0; i < n; i++) {
      pointCoords[2 * (i + clicksFromBox)] = boundaryClicks[i].x * modelScale.samScale;
      pointCoords[2 * (i + clicksFromBox) + 1] = boundaryClicks[i].y * modelScale.samScale;
      pointLabels[i + clicksFromBox] = boundaryClicks[i].clickType;
    }
    console.log(pointLabels, ' === pointLabels === ')

    // Add in the extra point/label when only clicks and no box
    // The extra point is at (0, 0) with label -1
    if (!clicksFromBox) {
      pointCoords[2 * n] = 0.0;
      pointCoords[2 * n + 1] = 0.0;
      pointLabels[n] = -1.0;
      // update n for creating the tensor
      n = n + 1;
    }

    // Create the tensor
    pointCoordsTensor = new Tensor("float32", pointCoords, [
      1,
      n + clicksFromBox,
      2,
    ]);
    pointLabelsTensor = new Tensor("float32", pointLabels, [
      1,
      n + clicksFromBox,
    ]);
  }
  const imageSizeTensor = new Tensor("float32", [
    modelScale.height,
    modelScale.width,
  ]);

  if (pointCoordsTensor === undefined || pointLabelsTensor === undefined)
    return;

  // If there is a previous tensor, use it, otherwise we default to an empty tensor
  const lastPredMaskTensor =
    last_pred_mask && clicks && !isFirstClick(clicks)
      ? last_pred_mask
      : new Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]);

  // +!! is javascript shorthand to convert truthy value to 1, falsey value to 0
  const hasLastPredTensor = new Tensor("float32", [
    +!!(last_pred_mask && clicks && !isFirstClick(clicks)),
  ]);

  return {
    image_embeddings: imageEmbedding,
    point_coords: pointCoordsTensor,
    point_labels: pointLabelsTensor,
    orig_im_size: imageSizeTensor,
    mask_input: lastPredMaskTensor,
    has_mask_input: hasLastPredTensor,
  };
};

export { setParmsandQueryModel, modelData };
