import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

from data.utils_old import create_data_lists


def extract_largest_item_label(voc07_dir):
    ''' extracts labels according to largest item in image for finetuning detection model
     Cross referenced labels are correct from here: https://github.com/DhruvJawalkar/object-detection-using-pytorch/blob/master/csvs/largest_item_classifier.csv
     '''

    DATA_DIR = Path(voc07_dir) / 'loader'

    for type in ['TRAIN', 'VAL', 'TEST']:
        DATA_IMAGES = json.load((DATA_DIR / f'{type}_images.json').open())
        DATA_ANNOTATIONS = json.load((DATA_DIR / f'{type}_objects.json').open())

        annotations_dic = defaultdict(lambda: [])

        for i, im_annotations in enumerate(DATA_ANNOTATIONS):
            im_name = DATA_IMAGES[i]
            boxes = im_annotations['boxes']
            for n, label in enumerate(im_annotations['labels']):
                annotations_dic[im_name].append((boxes[n], label))

        largest_ann_dic = defaultdict(lambda: ())

        for im_name, ann in annotations_dic.items():
            largest_ann_dic[im_name] = sorted(annotations_dic[im_name], key=lambda x: x[0][2] * x[0][3], reverse=True)[0] or ()

        df = pd.DataFrame(data={'file_name': DATA_IMAGES,
                                'labels': [largest_ann_dic[file][1] for file in DATA_IMAGES],
                                'boxes': [largest_ann_dic[file][0] for file in DATA_IMAGES]})

        df.to_csv(DATA_DIR / f'{type}_largest_item_labels.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-dir', type=str, help='Path to VOC dataset directory')

    args = parser.parse_args()
    data_dir = args.voc_dir

    # create_data_lists(voc07_path=f'{data_dir}/VOC2007',
    #                   voc12_path=f'{data_dir}/VOC2012',
    #                   output_folder=f'{data_dir}/VOC2007/loader')

    extract_largest_item_label(f'{data_dir}/VOC2007')
