import numpy as np
import cv2

from metrics.iou import IoU
from data_processing.data_preprocessor import preprocess_image

def modify_img_shape(min_size, height, width):
    '''
    Input - 

    min_size - Length of the Shortest Side of the Output.
    height - Original height of the Image.
    width - Original width of the Image.


    Output - 

    img - (min_size, resized_other_side)
    '''

    new_height, new_width = 0, 0
    multiplier = float(min_size / min(height, width))
    if height <= width:
        new_height = min_size
        new_width = int(width * multiplier)
    else:
        new_width = min_size
        new_height = int(height * multiplier)
    return (new_height, new_width)


def calculate_regression_targets(gta, anchor):
    '''
    Input - 
    
    gta - Ground Truth Bounding Box.
    anchor - Anchor.

    Output - 
    (tx, ty, tw, th) - Regression Targets.
    Parameterise the coordinates using the formula described in the FRCNN paper.
    '''

    ### Center Coordinates of Ground Truth ###
    
    gt_cx = (gta[0] + gta[2]) / 2
    gt_cy = (gta[0] + gta[2]) / 2

    ### Center Coordinates of Anchor ###
    
    anchor_cx = (anchor[0] + anchor[2]) / 2
    anchor_cy = (anchor[0] + anchor[2]) / 2

    tx = (gt_cx - anchor_cx) / (anchor[2] - anchor[0])
    ty = (gt_cy - anchor_cy) / (anchor[3] - anchor[1])
    tw = np.log((gta[2] - gta[0]) / (anchor[2] - anchor[0]))
    th = np.log((gta[3] - gta[1]) / (anchor[3] - anchor[1]))

    return (tx, ty, tw, th)



def calculate_anchors(config, img, bbox, bbox_labels, get_feature_shape_function):
    '''
    Input - 
    
    config - Contains Anchor Generating Parameters.
    img - Input image from which anchors are generated.
    bbox - Bounding Boxes in the Image.
    bbox_labels - Class Labels of Bounding Boxes.
    get_feature_shape_function - Calculates the Features Shape of the Base Network.

    Output - 


    '''

    ### Feature Map Shape ###

    (height, width) = img.shape[:2]
    (output_width, output_height) = get_feature_shape_function(width, height)

    ### Anchor Scales and Anchor Ratios ###

    anchor_ratios = config.anchor_box_ratios
    anchor_scales = config.anchor_box_scales
    stride = config.rpn_stride
    n_anchors = len(anchor_scales) * len(anchor_ratios)
    n_anchor_ratios, n_anchor_scales = len(anchor_ratios), len(anchor_scales)


    ### Initialising Output Variables ###
    
    y_rpn_overlap = np.zeros((output_height, output_width, n_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, n_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, n_anchors * 4))

    ### Ground Truth Anchors ###

    '''
    gta - [XMIN, YMIN, XMAX, YMAX]
    '''
    n_bboxs = len(bbox)
    gta = np.asarray(bbox)
    print(gta.shape, gta.dtype)

    ### Anchor Metrics ###

    '''
    n_anchors_for_bbox - Number of Anchors for a Bounding Box.
    
    best_anchor_for_bbox - Position of Best Anchor in the Feature Map for a Bounding Box.
                         - Format = (pixel row, pixel col, anchor_ratio, anchor_scale)
    
    best_anchor_coord_for_bbox -  Coordinates of the Best Anchor for a Bounding Box.
                               -  Format = (XMIN, YMIN, XMAX, YMAX)
    
    best_anchor_regr_for_bbox - Parameterised Regression Targets of the Anchor Coordinates for a Bounding Box.
                              - Format = (tx, ty, tw, th)

    best_iou_for_bbox - IoU of the best anchor for a Bounding Box.
    '''

    n_anchors_for_bbox = np.zeros(n_bboxs, dtype = np.int32)
    best_anchor_for_bbox = np.ones((n_bboxs, 4), dtype = np.int32) * -1
    best_anchor_coord_for_bbox = np.zeros((n_bboxs, 4), dtype = np.int32)
    best_anchor_regr_for_bbox = np.zeros((n_bboxs, 4), dtype = np.float32)
    best_iou_for_bbox = np.zeros(n_bboxs, dtype = np.float32)

    '''
    1. Find Best Anchor for Each BBOX.
    2. Classify it as 'Positive', 'Negative', 'Neutral' based on IoU.
    3. Each BBOX should have one anchor for it.
    4. The Anchor Coordinates are parameterised and used as a Label.
    5. Divide labels into Positive and Negative.
    6. Create Classification Labels.
    7. Create Regression Targets.
    '''

    for anchor_scale_idx in range(n_anchor_scales):
        for anchor_ratio_idx in range(n_anchor_ratios):
            anchor_x = anchor_scales[anchor_scale_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_scales[anchor_scale_idx] * anchor_ratios[anchor_ratio_idx][1]
            print(anchor_x, anchor_y)

            ### Anchor Coordinates - anchor_x1, anchor_y1, anchor_x2, anchor_y2 ###

            for ix in range(output_width):

                anchor_x1 = stride * (ix + 0.5) - anchor_x / 2
                anchor_x2 = stride * (ix + 0.5) + anchor_x / 2

                if anchor_x1 < 0 or anchor_x1 > width:
                    continue
                
                for iy in range(output_height):

                    anchor_y1 = stride * (iy + 0.5) - anchor_y / 2
                    anchor_y2 = stride * (iy + 0.5) + anchor_y / 2

                    if anchor_y1 < 0 or anchor_y1 > height:
                        continue
                    
                    bbox_type = 'neg'
                    best_iou_for_anchor = 0.0

                    for bbox_idx in range(n_bboxs):
                        current_iou_for_anchor = IoU(gta[bbox_idx], [anchor_x1, anchor_y1, anchor_x2, anchor_y2])

                        if current_iou_for_anchor > best_iou_for_bbox[bbox_idx]:
                            (tx, ty, tw, th) = calculate_regression_targets(gta[bbox_idx], [anchor_x1, anchor_y1, anchor_x2, anchor_y2])

                        if bbox_labels[bbox_idx] != config.class_mapping["background"]:

                            if current_iou_for_anchor > best_iou_for_bbox[bbox_idx]:
                                best_iou_for_bbox[bbox_idx] = current_iou_for_anchor
                                best_anchor_for_bbox[bbox_idx] = [iy, ix, anchor_ratio_idx, anchor_scale_idx]
                                best_anchor_coord_for_bbox[bbox_idx] = [anchor_x1, anchor_y1, anchor_x2, anchor_y2]
                                best_anchor_regr_for_bbox[bbox_idx] = [tx, ty, tw, th]

                            if current_iou_for_anchor > config.rpn_max_overlap:
                                bbox_type = 'pos'
                                n_anchors_for_bbox[bbox_idx] += 1

                                if current_iou_for_anchor > best_iou_for_anchor:
                                    best_iou_for_anchor = current_iou_for_anchor
                                    best_regr_for_anchor = (tx, ty, tw, th)

                            elif config.rpn_min_overlap < current_iou_for_anchor < config.rpn_max_overlap:                                
                                bbox_type = 'neutral' if bbox_type != 'pos' else bbox_type
                                # print("{} is neutral".format(current_iou_for_anchor))
                
                    anchor_idx = anchor_ratio_idx + n_anchor_ratios * anchor_scale_idx
                    anchor_regr_idx = 4 * anchor_idx

                    if bbox_type == 'neg':
                        y_is_box_valid[iy, ix, anchor_idx] = 1
                        y_rpn_overlap[iy, ix, anchor_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[iy, ix, anchor_idx] = 1
                        y_rpn_overlap[iy, ix, anchor_idx] = 1
                        y_rpn_regr[iy, ix, anchor_regr_idx: anchor_regr_idx + 4] = best_regr_for_anchor
                    elif bbox_type == 'neutral':
                        y_is_box_valid[iy, ix, anchor_idx] = 0
                        y_rpn_overlap[iy, ix, anchor_idx] = 0

    for bbox_idx in range(n_anchors_for_bbox.shape[0]):
        if n_anchors_for_bbox[bbox_idx] == 0:

            if best_anchor_for_bbox[bbox_idx, 0] == -1:
                continue
            
            (iy, ix, anchor_ratio_idx, anchor_scale_idx) = best_anchor_for_bbox[bbox_idx]
            anchor_idx = anchor_ratio_idx + n_anchor_ratios * anchor_scale_idx
            anchor_regr_idx = 4 * anchor_idx
            
            y_is_box_valid[iy, ix, anchor_idx] = 1
            y_rpn_overlap[iy, ix, anchor_idx] = 1
            y_rpn_regr[iy, ix, anchor_regr_idx:anchor_regr_idx + 4] = best_anchor_regr_for_bbox[bbox_idx]

    ### Convert Output Parameters ###
    # (output_height, output_width, n_anchors) -> (1, n_anchors, output_height, output_width) #

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))    
    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))

    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis = 0)
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis = 0)    
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis = 0)

    ### Positive and Negative Labels ###

    pos_indices = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1)) 
    neg_indices = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1)) 
    
    (n_positive, n_negative) =  (len(pos_indices[0]),len(neg_indices[0]))
    n_max_samples = config.max_samples

    ### Disabling Extra Indices ###

    if n_positive > n_max_samples/2:
        invalid_indices = np.random.choice(range(n_positive), n_max_samples/2, replace = False)
        y_is_box_valid[0, pos_indices[0][invalid_indices], pos_indices[1][invalid_indices], pos_indices[2][invalid_indices]] = 0
        n_positive = n_max_samples/2

    if n_negative + n_positive > n_max_samples:
        invalid_indices = np.random.choice(range(n_negative), n_negative - n_positive, replace = False)
        y_is_box_valid[0, neg_indices[0][invalid_indices], neg_indices[1][invalid_indices], neg_indices[2][invalid_indices]] = 0
        n_negative = n_positive

    ### Creating Classification and Regression Labels ###

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis = 1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis = 1), y_rpn_regr], axis = 1)

    '''
    print(y_is_box_valid.shape)
    print(y_rpn_overlap.shape)
    print(y_rpn_cls.shape)
    print(y_rpn_regr.shape)
    '''

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr), n_positive


def get_anchor_gt(config, all_img_data, mode = 'train'):
    '''
    Input - 
    
    img_data - Contains Image Data.
    mode - ['train', 'test'] - Tells whether to augument data or not.

    Output - 

    X - input image.
    Y - [classification scores, regression values].

    Steps - 

    1. Resize Image and Bounding Boxes to the sizes used in the original paper.
    2. Using the last feature map, generate the anchors.
    3. Preprocess the Image.
    4. Scale the Regression Targets as done in the Research Paper.
    5. Make the Labels Backend Compatible.

    '''

    base_network = None
    if config.base_network is 'vgg16':
        from networks import vgg16 as base_network
        get_feature_shape_function = base_network.get_output_shape

    while True:

        for img_data in all_img_data:
            print(img_data)
            x_img = cv2.imread(img_data["filepath"])
            (height, width) = x_img.shape[:2]
            (rows, cols) = img_data["height"], img_data["width"]

            ### Check Image Dimensions ###

            assert height == rows
            assert width == cols

            ### Resizing Image ###

            min_size = config.img_size

            (new_height, new_width) = modify_img_shape(min_size, height, width)

            x_img = cv2.resize(x_img, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

            ### Resizing Bounding Boxes ###
            
            print(img_data["bbox"])
            height_multiplier, width_multiplier = float(new_height/height), float(new_width/width)
            multiplier = [width_multiplier, height_multiplier] * 2

            ### Bounding Boxes ###

            bbox = np.asarray([box[:4] for box in img_data["bbox"]], dtype = np.float32)
            bbox *= multiplier
            bbox = bbox.astype(np.int32)

            ### Class for each Bounding Box ###

            bbox_labels = [config.class_mapping[box[4]] for box in img_data["bbox"]]

            ### Generating Anchors From Given Feature Map Shape and Boudning Boxes of an Image ###
            
            (y_rpn_cls, y_rpn_regr, n_positive) = calculate_anchors(config, x_img, bbox, bbox_labels, get_feature_shape_function)

            ### Preprocess Image ###

            x_img = preprocess_image(config, x_img)
            x_img = np.transpose(x_img, (2, 0, 1))
            x_img = np.expand_dims(x_img, axis = 0)

            ### Apply Scaling to Regression Targets ###

            y_regr_start_idx = y_rpn_regr.shape[1]//2
            y_rpn_regr[:, y_regr_start_idx:, :, :] *= config.std_scaling

            ### Convert Model Inputs to Backend Compatible Shape ###

            if config.backend == 'tensorflow':
                '''
                Tensorflow Accepts (1, height, width, depth)
                NOTE: Check that [0](2, 0, 1) is inverse of (0, 2, 3, 1).
                '''

                x_img = np.transpose(x_img, (0, 2, 3, 1))
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))
            
            yield np.copy(x_img),[np.copy(y_rpn_cls), np.copy(y_rpn_regr)]