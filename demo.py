import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torchvision import transforms
from pytorch_model.yolo import YoloV2
from pytorch_model.yoloUtilTorch import yolo_head, yolo_eval
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision.utils import make_grid, save_image
import argparse
import colorsys
import imghdr
import os
import random

parser = argparse.ArgumentParser(description='The YOLO Test Phase on pretrained model')
parser.add_argument('model_path', help='path to pickle model file containing pretrained weight')
parser.add_argument('-a','--anchors_path',help='path to anchors file',default='model_data/yolo_anchors.txt')
parser.add_argument('-c','--classes_path',help='path to classes file', default='model_data/coco_classes.txt')
parser.add_argument('-cu','--cuda',help='Flag to use cuda',default=False)
parser.add_argument('-t','--test_path',help='path to directory of test images, defaults to images/',default='images')
parser.add_argument('-o','--output_path',help='path to output test images, defaults to images/out',default='images/out')
parser.add_argument('-s','--score_threshold',type=float,help='threshold for bounding box scores, default .3',default=.3)
parser.add_argument('-iou','--iou_threshold',type=float,help='threshold for non max suppression IOU, default .5',default=.5)

# Define Data Transformations
data_tarnsform = transforms.Compose([transforms.Scale([416,416]),transforms.ToTensor()])


def _main(args):

    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.p'), 'Pretrained model must be a .p file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    test_path = os.path.expanduser(args.test_path)
    output_path = os.path.expanduser(args.output_path)

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    # Read Class Names
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    # Read Prior Knowledge about anchors size
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    # Create Architecture and load pretrained model
    yolo = YoloV2(model_path)
    # Move Batchnorm to evaluation state
    yolo.eval()
    # Print Yolo Architecture
    print(yolo)

    torch.manual_seed(0)
    if args.cuda:
        torch.cuda.manual_seed_all(0)
        yolo = nn.DataParallel(yolo, device_ids=[0,1])
        FloatType = torch.cuda.FloatTensor
        LongType = torch.cuda.LongTensor
        yolo = yolo.cuda()
        cudnn.benchmark = True
    else:
        FloatType = torch.FloatTensor
        LongType = torch.LongTensor

    num_classes = len(class_names)
    num_anchors = len(anchors)
    anchors = autograd.Variable(torch.from_numpy(anchors))

    model_image_size = (416,416)
    is_fixed_size = model_image_size != (None, None)


    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    for image_file in os.listdir(test_path):
        try:
            image_type = imghdr.what(os.path.join(test_path, image_file))
            if not image_type:
                continue
        except IsADirectoryError:
            continue
        image = Image.open(os.path.join(test_path, image_file))
        inputs = data_tarnsform(image).unsqueeze(0)
        output = yolo(autograd.Variable(inputs,volatile=True))
        yolo_outputs = yolo_head(output, anchors, len(class_names), FloatType, LongType)
        input_image_shape = [image.size[1], image.size[0]]
        out_boxes, out_scores, out_classes = yolo_eval(
            yolo_outputs,
            input_image_shape, 
            FloatType, 
            LongType,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold)
        print('Found {} boxes for {}'.format(len(out_boxes), image_file))
        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        image.save(os.path.join(output_path, image_file), quality=90)

if __name__ == '__main__':
    _main(parser.parse_args())
