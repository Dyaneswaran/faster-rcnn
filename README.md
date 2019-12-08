# Faster-RCNN
> A Keras Implementation of Faster-RCNN using a Tensorflow Backend for Custom Object Detection.

This is a work-in-progress. As of now, only RPN training is feasible.

This implementation works only in the following environment.
> 1. Keras 2.2.4 with Tensorflow Backend.
> 2. Tensorflow 1.14.0.

## Steps Involved in Creating a FRCNN Object-Detection Model

- Generating Region Proposals (RPN)
- Non-Maximum Suppression (NMS)
- ROI Pooling
- RCNN for Classification and Adjusting Bounding Box Regression Values


## Training RPN

Use the following command to train the RPN.

`python train_rpn.py --input_path serengeti --dataset_name serengeti --network vgg16 --n_epochs 5 --n_epoch_length 10`
                    
**NOTE** - I have attached as much explanation as possible in the code. Hopefully, its understandable.
Cheerio.

### TO-DO
>
- Data Augmentation
- Validation Data in RPN
- Non-Maximum Suppression
- ROI Pooling
- RCNN
- Support for Other Base Networks as Backbone
