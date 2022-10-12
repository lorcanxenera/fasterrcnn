import os
import random

import torch
from tqdm import tqdm

from .config import Config
from .dataset import myDataset
from .helpers import COCO_INSTANCE_CATEGORY_NAMES, dataHelper

# Prepare project root filepath
project_name = ''
root_dir = os.path.join('', project_name)

# Remaining filepaths
main_folder = os.path.join(root_dir, 'main')
main_dir = os.path.join(main_folder, 'images')
main_filenames = [os.path.join(main_dir, filename) for filename in os.listdir(main_dir)]
test_image = random.choice(main_filenames)
annotations_dir = os.path.join(main_folder, 'annotations')
test_annots_file, train_annots_file, train_test_annots_file = os.listdir(annotations_dir)
background_dir = os.path.join(root_dir, 'background')

# Prepare Config, Helper, and Dataset objects
print("Torch version:", torch.__version__)
helper = dataHelper()
config = Config(main_dir, annotations_dir, train_annots_file)
# Create Dataset
my_dataset = myDataset(
    root=config.train_data_dir, annotation=config.train_coco, transforms=helper.get_transform()
)

# Prepare DataLoader
data_loader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=config.train_batch_size,
    shuffle=config.train_shuffle_dl,
    num_workers=config.num_workers_dl,
    collate_fn=helper.collate_fn,
)

# Select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Create model
model = helper.get_model_instance_segmentation(config.num_classes)

# Move model to the right device
model.to(device)

# Parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(
    params, lr=config.lr, weight_decay=config.weight_decay
)

len_dataloader = len(data_loader)

# Train model
for epoch in tqdm(range(config.num_epochs)):
    print(f"Epoch: {epoch}/{config.num_epochs}")
    model.train()
    i = 0
    for imgs, annotations in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i%100 == 0:
            print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}")

save_path = os.path.join(main_folder, os.pardir, 'model.pth')
torch.save(model.state_dict(), save_path)