import os
import cv2
import re


def parse_file(input_dir, dataset_name):
    '''
    Parses an annotation file.
    Input - Input Directory, Dataset Name
    Output - Image Data in the form of a dictionary.
    '''

    ### STRUCTURE OF AN ANNOTATION ###
    '''
    (image_name, XMIN, YMIN, XMAX, YMAX, class)
    '''

    ### THINGS THAT WE NEED TO FIND OUT FROM THE DATA ###

    '''
    1. Class Count.
    2. Class Mapping.
    3. Image Data - (Width, Height, Bounding Boxes, Class).

    NOTE: Background class is always last. Even if there is not one to begin with.  
    '''

    data = {}
    class_mapping = {}
    class_count = {}
    exists_background = False

    filename = os.path.join(input_dir, dataset_name) + ".csv"

    with open(filename, "r") as f:
        for line in f:
            (img_name, xmin, ymin, xmax, ymax, classname) = re.sub(r"\s+", "", line, flags=re.UNICODE).lower().split(",")
            
            if img_name not in data:
                data[img_name] = {}
                data[img_name]["bbox"] = []

            data[img_name]["image_name"] = img_name
            
            img_filepath = os.path.join(input_dir, "images", img_name)
            data[img_name]["filepath"] = img_filepath

            try:
                img = cv2.imread(img_filepath)
                data[img_name]["height"] = img.shape[0]
                data[img_name]["width"] = img.shape[1]
            except:
                print(os.path.join(input_dir, "images", img_name))
                print("{} read error. Please check if file exists.".format(img_name))
                data[img_name]["height"] = 0
                data[img_name]["width"] = 0

            if classname not in class_count:
                exists_background = True if classname is "background" else False
                class_count[classname] = 1
                class_mapping[classname] = len(class_mapping)
            else:
                class_count[classname] += 1

            data[img_name]["bbox"].append([int(xmin), int(ymin), int(xmax), int(ymax), classname])

    if exists_background:
        if class_mapping["background"] != len(class_mapping)-1:
            #Swap last class with background.
            last_key = [k for k in class_mapping if class_mapping[k] == len(class_mapping) - 1][0]
            class_mapping["background"], class_mapping[last_key] = class_mapping[last_key], class_mapping["background"]
    else:
        #Create a Background class explicitly if not present.
        class_count["background"] = 0
        class_mapping["background"] = len(class_mapping)

    data = [ data[key] for key in data.keys()]

    return class_count, class_mapping, data

def get_data(input_dir, dataset_name):
    if not os.path.exists(input_dir):
        print("{} doesnot exist. Please give a valid input directory.".format(input_dir))
        return
    
    dataset_labels_file = os.path.join(input_dir, dataset_name) + ".csv"
    if not os.path.exists(dataset_labels_file):
        print("{} annotation file doesnot exits. Please create one with {} as name.".format(dataset_labels_file, dataset_name))

    print("Parsing Annotations File....")
    class_count, class_mapping, img_data = parse_file(input_dir, dataset_name)
    return class_count, class_mapping, img_data