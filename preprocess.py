# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from PIL import Image
import numpy as np

def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def transform(in_file):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = Image.open(in_file).convert('RGB')
    w = img.size[0]
    h = img.size[1]
    if w > h:
        input_size = np.array([256, 256 * w / h])
    else:
        input_size = np.array([256 * h / w, 256])
    input_size = input_size.astype(int)

    img = resize(img, input_size)
    img = np.array(img, dtype=np.float32)
    img = center_crop(img, 224, 224)
    img = img / 255.
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img[..., 0] /= std[0]
    img[..., 1] /= std[1]
    img[..., 2] /= std[2]

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return img


def main():
    img = transform("./test.jpg")
    img.tofile("./test.bin")
    print("success in preprocess")

if __name__ == "__main__":
    main()