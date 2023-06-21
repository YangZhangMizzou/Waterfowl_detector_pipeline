from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training import models


dataset_params = {
    'data_dir':'./dataset/drone_collection_height',
    'train_images_dir':'train/15/small_only/images',
    'train_labels_dir':'train/15/small_only/labels',
    'val_images_dir':'test/15/small_only/images',
    'val_labels_dir':'test/15/small_only/labels',
    'test_images_dir':'test/15/small_only/images',
    'test_labels_dir':'test/15/small_only/labels',
    'classes': ['Bird']
}

CHECKPOINT_DIR = './checkpoints'
trainer = Trainer(experiment_name='waterfowl_train', ckpt_root_dir=CHECKPOINT_DIR)
model = models.get('yolo_nas_l', num_classes=len(dataset_params['classes']), pretrained_weights="coco")