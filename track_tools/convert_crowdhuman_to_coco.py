"""
https://github.com/xingyizhou/CenterTrack
Modified by Peize Sun
"""

import os
import numpy as np
import json
from PIL import Image

DATA_PATH = 'crowdhuman'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
SPLITS = ['val', 'train']
DEBUG = False

def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = os.path.join(DATA_PATH, split)
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
        #ann_path = DATA_PATH + 'annotation_{}.odgt'.format(split)
        #anns_data = load_func(ann_path)
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0


        for seq in sorted(seqs):
            image_cnt += 1
            seq_path = os.path.join(data_path, seq)
            #file_path = DATA_PATH + 'CrowdHuman_{}/'.format(split) + '{}.jpg'.format(ann_data['ID'])
            im = Image.open(seq_path)
            image_info = {'file_name': seq,
                          'id': image_cnt,
                          'height': im.size[1], 
                          'width': im.size[0]}
            out['images'].append(image_info)

            if split == "val":
                ann_path = os.path.join(seq_path.replace('val', 'labels_with_ids'), seq.replace("jpg","txt"))
            else:
                ann_path = os.path.join(seq_path.replace('train', 'labels_with_ids'), seq.replace("jpg","txt"))


            if split != 'test':
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',', usecols=range(9))
                for i in range(anns.shape[0]):
                    ann_cnt += 1
                    ann = {'id': ann_cnt,
                         'category_id': 1,
                         'image_id': image_cnt,
                         'bbox_vis': anns[i][2:6].tolist(),
                         'bbox': anns[i][2:6].tolist(),
                         'area': float(anns[i][4] * anns[i][5]),
                         'iscrowd': 0}

                    out['annotations'].append(ann)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
