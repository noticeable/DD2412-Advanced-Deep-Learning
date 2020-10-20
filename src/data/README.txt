If you downlaod the dataset from Kaggle

1 - create a folder called data
2 - mv VOC2012/JPEGImages data/images
3 - mv VOC2012/SegmentationClass data/labels

we have two transformation
input_transform will be applied on images (inputs)
target_transform will be applied on targes (mask or lables)



In the future we need to have 3 such folders, i.e. train, validation and test