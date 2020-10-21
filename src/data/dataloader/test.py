from datasets import PascalVOCDataset
from utils import *

# Data parameters
data_folder = r'E:\VOC\loader'  # folder with data files
keep_difficult = False  # use objects considered difficult to detect?
batch_size = 8
workers = 0  # number of workers for loading data in the DataLoader
# Custom dataloaders
train_dataset = PascalVOCDataset(data_folder,
                                    split='train',
                                    keep_difficult=keep_difficult)
# Custom dataloaders
val_dataset = PascalVOCDataset(data_folder,
                                    split='val',
                                    keep_difficult=keep_difficult)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                            collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

# Batches
for i, (images, boxes, labels, _) in enumerate(train_loader):
    print(1)



    