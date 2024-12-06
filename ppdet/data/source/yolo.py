# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np

from ppdet.core.workspace import register, serializable

from .dataset import DetDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

from .readAnno import readAnno
from PIL import Image
from tqdm import tqdm

def load_all_datas(txt_paths):
    datas = {}
    txt_paths = txt_paths if isinstance(txt_paths, list) else [txt_paths]
    for txt_path in txt_paths:
        with open(txt_path, 'r', encoding='utf-8') as f:
            alllines = f.readlines()
        for lineId in range(len(alllines)//2):
            img_path = alllines[2*lineId].strip()
            if not img_path:
                continue
            anno_path = alllines[2*lineId+1].strip()
            if not anno_path:
                continue
            datas[img_path] = anno_path
    return datas  

@register
@serializable
class YOLODataSet(DetDataset):
    """
    Load dataset with PascalVOC format.

    Notes:
    `anno_path` must contains xml file and image file path for annotations.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): voc annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        label_list (str): if use_default_label is False, will load
            mapping between category and class index.
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total 
            record's, if empty_ratio is out of [0. ,1.), do not sample the 
            records and use all the empty entries. 1. as default
        repeat (int): repeat times for dataset, use in benchmark.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 label_list=None,
                 allow_empty=False,
                 empty_ratio=1.,
                 repeat=1):
        super(YOLODataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            repeat=repeat)
        self.label_list = label_list
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio    
        self.img_datas = [] 

    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = random.sample(records, sample_num)
        return records

    def parse_dataset(self, ):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        # mapping category name to class id
        # first_class:0, second_class:1, ...
        cname2cid = {}
        label_path = os.path.join(self.dataset_dir, self.label_list)
        if not os.path.exists(label_path):
            raise ValueError("label_list {} does not exists".format(
                label_path))
        with open(label_path, 'r') as fr:
            label_id = 0
            for line in fr.readlines():
                cname2cid[line.strip()] = label_id
                label_id += 1

        # 读取数据集
        records = []
        empty_records = []
        ct = 0
        alldatas = load_all_datas(anno_path)
        print("读取数据集：")
        for img_file, label_file in tqdm(alldatas.items()):
            boxes = readAnno(label_file, list(cname2cid.keys()))
            num_bbox, i = len(boxes), 0
            gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
            gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_score = np.zeros((num_bbox, 1), dtype=np.float32)
            # 打开图片文件
            img = Image.open(img_file)
            # 获取图片的宽和高
            im_w, im_h = img.size

            for box in boxes:
                cls_id, x, y, w, h = box
                x1, y1 = (x-w/2)*im_w, (y-h/2)*im_h
                x2, y2 = x1+w*im_w, y1+h*im_h
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(im_w - 1, x2)
                y2 = min(im_h - 1, y2)
                
                gt_bbox[i, :] = [x1, y1, x2, y2]
                gt_class[i, 0] = cls_id
                gt_score[i, 0] = 1.
                i += 1
               
            gt_bbox = gt_bbox[:i, :]
            gt_class = gt_class[:i, :]
            gt_score = gt_score[:i, :]

            im_id = np.array([ct])
            rec_data = {
                'im_file': img_file,
                'im_id': im_id,
                'h': im_h,
                'w': im_w
            } if 'image' in self.data_fields else {}
            gt_rec = {
                'gt_class': gt_class,
                'gt_score': gt_score,
                'gt_bbox': gt_bbox,
            }
            img_data = rec_data.copy()
            img_data.update(gt_rec.copy())
            for k, v in gt_rec.items():
                if k in self.data_fields:
                    rec_data[k] = v

            if len(boxes) == 0:
                empty_records.append(rec_data)
            else:
                records.append(rec_data)
            self.img_datas.append(img_data)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any record in %s' % (self.anno_path)
        logger.debug('{} samples in file {}'.format(ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs, self.cname2cid = records, cname2cid

    def get_label_list(self):
        return os.path.join(self.dataset_dir, self.label_list)

