import json

def get_label2category_map(label_map_file='../weights/label_map.json'):
    # get label2category map to save category with image
    with open(label_map_file, 'r') as f:
        category2labelmap = json.load(f)
        label2categorymap = {category2labelmap[category]: category for category in category2labelmap}
        return label2categorymap