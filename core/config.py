import os

class Config():
    def __init__(self, main_dir, annotations_dir, train_annots_file):

        self.train_data_dir = main_dir
        self.train_coco = os.path.join(annotations_dir, train_annots_file)

        # Batch size
        self.train_batch_size = 1

        # Params for dataloader
        self.train_shuffle_dl = True
        self.num_workers_dl = 2

        # Params for training

        ######################################
        ######################################
        ######################################
        # SET THE NUMBER OF CLASSES AND EPOCHS
        self.num_classes = 2
        self.num_epochs = 1

        self.lr = 0.0005
        self.momentum = 0.9
        self.weight_decay = 0.005