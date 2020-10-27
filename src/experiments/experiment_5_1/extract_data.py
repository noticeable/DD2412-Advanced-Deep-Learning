import json
from pathlib import Path
import random
import pandas as pd
from experiments.shared import get_label2category_map


def extract_images(data_dir, experiment_data_fpath, n_images=20):
    DATA_DIR = Path(data_dir)

    for type in ['TRAIN', 'VAL', 'TEST']:
        DATA_IMAGES = json.load((DATA_DIR / f'{type}_images.json').open())
        DATA_ANNOTATIONS = json.load((DATA_DIR / f'{type}_objects.json').open())

        labels = []
        file_names = []
        categories = []

        for i, im_annotations in enumerate(DATA_ANNOTATIONS):
            im_labels = im_annotations['labels']
            if len(im_labels) == 2 and im_labels[0] != im_labels[1]: # only get those with 2, distinct annotated categories
                labels.append(im_labels)
                im_categories = []
                for label in im_labels:
                    im_categories.append(label2categorymap[label])
                categories.append(im_categories)
                im_name = DATA_IMAGES[i]
                file_names.append(im_name)

        to_sample = random.sample(range(len(file_names)), n_images)
        images_sampled = [file_names[i] for i in to_sample]
        labels_samples = [labels[i] for i in to_sample]
        categories_sampled = [categories[i] for i in to_sample]


        df = pd.DataFrame(data={'images': images_sampled,
                                'labels': labels_samples,
                                'categories': categories_sampled})

        df.to_csv(experiment_data_fpath, index=False)


if __name__ == '__main__':
    label2categorymap = get_label2category_map()
    experiment_data_fpath = 'experiments/experiment_5_1/experiment_data.csv'
    data_dir = '../datasets/VOC/VOC2007/loader'
    extract_images(data_dir, experiment_data_fpath)
