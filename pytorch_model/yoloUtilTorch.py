import torch
import torch.nn.functional as F
from .box_utils import nms
import torch.autograd as auto
import math
import pdb

def yolo_loss(args,
            anchors,
            num_classes,
            epoch, 
            index,
            FloatType, 
            LongType,
            rescroe_confidence = False,
            print_loss=True):
    """YOLO localization loss function.

    Parameters
    ----------
    yolo_output : variable-tensor
        Final convolutional layer features.

    true_boxes : variable-tensor
        Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
        containing box x_center, y_center, width, height, and class.

    detectors_mask : variable-tensor
        0/1 mask for detector positions where there is a matching ground truth.

    matching_true_boxes : variable-tensor
        Corresponding ground truth boxes for positive detector positions.
        Already adjusted for conv height and width.

    anchors : variable-tensor
        Anchor boxes for model.

    num_classes : int
        Number of object classes.

    rescore_confidence : bool, default=False
        If true then set confidence target to IOU of best predicted box with
        the closest matching ground truth box.

    print_loss : bool, default=False
        If True then use a tf.Print() to print the loss components.

    Returns
    -------
    mean_loss : float
        mean localization loss across minibatch
    """
    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args
    num_anchors = len(anchors)
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1
    # 0 <= pred_xy <= 1 : normalized value, shape = (bs, H, W, A, 2)
    # 0 <= pred_wh : normalized value, shape = (bs, H, W, A, 2)
    # 0 <= pred_confidence <= 1 (scaled with sigmoid operation), shape = (bs, H, W, A, 1)
    # pred_class_prob : applied softmax along the last dimension, shape = (bs, H, W, A, 80)
    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(
        yolo_output,
        anchors,
        num_classes, 
        FloatType, 
        LongType)

    yolo_output = yolo_output.permute(0,2,3,1)
    # yolo_output_shape = (bs, H, W, (4 + 1 + 80) * B)
    yolo_output_shape = yolo_output.size()
    
    # reshape yolo output to = (bs, H, W, B, 4 + 1 + 80)
    feats = yolo_output.contiguous().view(-1, yolo_output_shape[1],yolo_output_shape[2], num_anchors ,num_classes + 5)
    
    # used directly in loss function
    pred_boxes = torch.cat([sigmoid(feats[..., 0:2]), feats[..., 2:4]], dim=-1)

    # In the following, we are going to select the best anchor matched with our ground truth
    # So, we need to compare the Ground Truth (true_boxes) with the pred_xy, 
    # pred_wh, pred_confidence, pred_class_prob. 
    
    # pred_xy new shape for broadcasting: (bs, H, W, A, 1, 2)
    pred_xy = pred_xy.unsqueeze(4)
    # pred_wh new shape for broadcasting: (bs, H, W, A, 1, 2)
    pred_wh = pred_wh.unsqueeze(4)

    # find the up-left and bottom-right corners of the predicted bounding boxes
    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    # true_boxes_shape = (bs, max_bboxes, 6)
    true_boxes_shape = true_boxes.size()
    
    # reshape the true_boxes matrix to (bs, 1, 1, 1, max_boxes, 6)
    true_boxes = true_boxes.contiguous().view(true_boxes_shape[0],
        1,1,1, true_boxes_shape[1],
        true_boxes_shape[2])

    # true_xy shape is (bs, 1, 1, 1, max_boxes, 2)
    true_xy = true_boxes[..., 0:2]
    # true_wh shape is (bs, 1, 1, 1, max_boxes, 2)
    true_wh = true_boxes[..., 2:4]

    # find the up-left and bottom-right corners of the Ground truthes
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half
    

    intersect_mins = torch.max(pred_mins, auto.Variable(true_mins))
    intersect_maxes = torch.min(pred_maxes, auto.Variable(true_maxes))

    intersect_wh = torch.max(intersect_maxes - intersect_mins,
                 auto.Variable(torch.Tensor([0.]).type(FloatType)))
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    
    union_areas = pred_areas + auto.Variable(true_areas) - intersect_areas
    iou_scores = intersect_areas / union_areas

    best_ious, _ = iou_scores.max(dim=4)
    object_detections = (best_ious > 0.6).type_as(iou_scores)

    no_object_weights = (no_object_scale * (1 - object_detections) * (1 - detectors_mask.squeeze(-1)))
    no_objects_loss = no_object_weights * ((-pred_confidence) ** 2) 

    if rescroe_confidence:
        objects_loss = (object_scale * detectors_mask * ((best_ious-pred_confidence) ** 2))
    else:
        objects_loss = (object_scale * detectors_mask * ((1-pred_confidence) ** 2))

    confidence_loss = objects_loss + no_objects_loss

    # matching_classes shape = (bs, H, W, A, 5)
    matching_classes = matching_true_boxes[..., 4].type(LongType)
    # matching_classes new shape = (bs, H, W, A, 5, 1)
    matching_classes = matching_classes.unsqueeze(-1)
    # matching_classes new shape = (bs, H, W, A, 5, 80)
    matching_classes = auto.Variable(one_hot_along_dim(matching_classes, num_classes))

    classification_loss = (class_scale * detectors_mask * ((matching_classes - pred_class_prob) ** 2).sum(dim = -1).unsqueeze(-1))

    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = (coordinates_scale * detectors_mask * ((matching_boxes - pred_boxes) ** 2).sum(dim = -1).unsqueeze(-1))
    
    confidence_loss_sum = torch.sum(confidence_loss)
    classification_loss_sum = torch.sum(classification_loss)
    coordinates_loss_sum = torch.sum(coordinates_loss)

    total_loss = 0.5 * (confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)
    

    if print_loss:
        print("Epoch={}, Bacth={}, yolo_loss={}, conf_loss={}, class_loss={}, box_coord_loss={}, small_Object_loss={}".format(epoch, index, total_loss.data[0], 
            confidence_loss_sum.data[0], classification_loss_sum.data[0], coordinates_loss_sum.data[0], loss_small.data[0]))
    return total_loss


def preprocess_true_boxes(true_boxes, anchors, image_size, FloatType, subsample_d):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : torch array
        List of ground truth boxes in form of relative x, y, w, h, class, smallness.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : torch array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    smallObject_mask: 
        Same shape as detectors_mask with the corresponding smallness flag
        for each bounding boxes.
    """
    height, width = image_size
    num_anchors = len(anchors)

    assert height % subsample_d == 0     # Downsampling scale cheaking
    assert width % subsample_d == 0      # Downsampling scale cheaking

    conv_height = height // subsample_d
    conv_width = width // subsample_d
    anchors = (anchors/13)*conv_height

    num_box_params = true_boxes.size()[1] - 1   # number of parameters for each boxes without smallness param.
    
    detectors_mask = torch.zeros(
        conv_height, conv_width, num_anchors, 1).type(FloatType)
    
    smallObject_mask = torch.zeros(
        conv_height, conv_width, num_anchors, 1).type(FloatType)
    
    matching_true_boxes = torch.zeros(
        conv_height, conv_width, num_anchors, num_box_params).type(FloatType)

    for box in true_boxes:
        box_class = box[4:5]
        smallness = int(box[5])
        box = box[0:4] * torch.Tensor([conv_width, conv_height, conv_width, conv_height]).type(FloatType)
        i = int(box[1]) # y
        j = int(box[0]) # x
        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = anchor / 2.
            anchor_mins = -anchor_maxes

            intersect_mins = torch.max(box_mins, anchor_mins)
            intersect_maxes = torch.min(box_maxes, anchor_maxes)
            intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.Tensor([0. ,0.]).type(FloatType))
            intersect_area = intersect_wh[0]*intersect_wh[1]
            box_area = box[2]*box[3]
            anchor_area = anchor[0]*anchor[1]
            iou = (intersect_area) / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k
        if best_iou > 0:
            detectors_mask[i,j,best_anchor] = 1
            if smallness == 1:
                smallObject_mask[i,j,best_anchor] = 1
            adjusted_box = torch.Tensor([box[0] - j,
                box[1] - i,
                math.log(box[2]/anchors[best_anchor][0]),
                math.log(box[3]/anchors[best_anchor][1]), 
                box_class[0]]).type(FloatType)
            matching_true_boxes[i,j,best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes, smallObject_mask





def yolo_eval(yolo_outputs,
              image_shape,
              FloatType, 
              LongType,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    boxes, scores, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)
    # Scale boxes back to original image shape.
    if boxes is not None:
        height = image_shape[0]
        width = image_shape[1]
        image_dims = torch.Tensor([[height, width, height, width]]).type(FloatType)
        image_dims = auto.Variable(image_dims.view(1, 4))
        boxes = boxes * image_dims
        keep, count = nms(boxes.data, scores.data, overlap=iou_threshold, top_k=max_boxes)
        
        boxes = boxes.data.cpu().numpy()[keep[0:count],:]
        scores = scores.data.cpu().numpy()[keep[0:count]]
        classes = classes.data.cpu().numpy()[keep[0:count]]
    return boxes, scores, classes

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return torch.cat((box_mins[..., 1:2],  # y_min
                            box_mins[..., 0:1],  # x_min
                            box_maxes[..., 1:2],  # y_max
                            box_maxes[..., 0:1]  # x_max
            ),dim = -1)  

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_class_scores, box_classes = box_scores.max(dim=-1)
    prediction_mask = box_class_scores >= threshold
    prediction_mask = prediction_mask.view(-1)

    if prediction_mask.data.nonzero().numel() != 0:
        indeces = prediction_mask.data.nonzero()[:,0]
        boxes = boxes.view(-1,4)[indeces]
        scores = box_class_scores.view(-1)[indeces]
        classes = box_classes.view(-1)[indeces]
    else:
        boxes = scores = classes = []
    return boxes, scores, classes      

# Broadcasting, Parallel
def yolo_head(feats, anchors, num_classes, FloatType, LongType):
    feats = feats.permute(0,2,3,1)
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors = anchors.view([1, 1, 1, num_anchors, 2]).type(FloatType)
    conv_dims = torch.Tensor([feats.size(1),feats.size(2)]).type(LongType)

    conv_height_index = torch.arange(0, conv_dims[0]).type(FloatType)
    conv_width_index = torch.arange(0, conv_dims[1]).type(FloatType)

    conv_height_index = conv_height_index.repeat(conv_dims[1])
    conv_width_index = conv_width_index.unsqueeze(0).repeat(conv_dims[0],1).t()
    conv_width_index = conv_width_index.contiguous().view(-1)
    conv_index = torch.stack((conv_height_index,conv_width_index)).t()
    conv_index = auto.Variable(conv_index.contiguous().view(1,conv_dims[0],conv_dims[1],1,2))
    feats = feats.contiguous().view(-1, conv_dims[0],
     conv_dims[1],num_anchors, 5 + num_classes)
    conv_dims = auto.Variable(conv_dims.contiguous().view(1,1,1,1,2).type_as(feats.data))

    box_xy = F.sigmoid(feats[..., :2])
    box_wh = torch.exp(feats[..., 2:4])
    box_confidence = F.sigmoid(feats[..., 4:5])
    box_class_probs = F.softmax(feats[..., 5:], dim = -1)

    # pdb.set_trace()
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = (box_wh * anchors) / conv_dims
    return box_xy, box_wh, box_confidence, box_class_probs

def sigmoid(inp):
    return 1./(1. + torch.exp(-1.*inp))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp(x - torch.max(x).expand_as(x))
    return e_x / e_x.sum().expand_as(x)

def one_hot(x, num_classes):
    vec = torch.zeros(num_classes).type_as(x.data)
    vec[x.data[0]] = 1
    return vec

def softmax_along_dim(inp, axis=0):
    batch, height, width, na, probs = inp.size()
    inp = inp.contiguous().view(-1,80)
    res = torch.stack([
                softmax(x_i) for i, x_i in enumerate(torch.unbind(inp, dim=axis), 0)
            ], dim=axis)
    return res.contiguous().view(batch, height,  width, na, probs)

def one_hot_along_dim(inp, num_classes, axis=0):
    batch, height, width, na, probs = inp.size()
    inp = inp.contiguous().view(-1,1)
    res = torch.stack([
                one_hot(x_i, num_classes) for i, x_i in enumerate(torch.unbind(inp, dim=axis), 0)
            ])
    return res.contiguous().view(batch, height,  width, na, num_classes)