import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



###################
###################
###################
# SET OBJECT LABELS
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', '']

# Ensure tensorised inputs
class dataHelper():
    def __init__():
        pass

    def get_transform(self):
        custom_transforms = []
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)


    # collate_fn needs for batch
    def collate_fn(self, batch):
        return tuple(zip(*batch))


    def get_model_instance_segmentation(self, num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model


    def get_prediction(self, img_path, confidence, model):
        # Open image
        img = Image.open(img_path)
        transform = self.get_transform()
        img = transform(img)
        # Prepare model for inference
        if model.training:
            model.eval()
        device = torch.device("cpu")
        model.to(device)
        # Perform inference
        pred = model([img])
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return pred_boxes, pred_class


    def detect_objects(self, img_path, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    # Return prediction objects
        boxes, pred_cls = self.get_prediction(img_path, confidence)
        # Annotate base image with bounding boxes
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Draw each box
        for i in range(len(boxes)):
            input0 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
            input1 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
            cv2.rectangle(img, input0, input1, color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], input0, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        # Display image
        plt.close('all')
        plt.figure(figsize=(20,30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()