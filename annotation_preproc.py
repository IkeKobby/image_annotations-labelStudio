# import some necessary libraries
import pandas as pd
import numpy as np
from skimage.io import imread
import cv2
import os

def updateAnnotations(image_directory,
                      target_image_directory,
                      annotation_data,
                      newImageSize = (300, 300)):
    """
    A preprocessing helper method that resizes annotated images and
    updates `segmentation` and `bbox`. 
    Args: 
        `image_directory`; the directory/folder to the annotated images exported from `labelstudio`.

        'target_image_directory`; the new path where you wish to save your preprocessed image, that is,
                                the resized image. Also, note that, if your images are already 3 channel 
                                images then you'd need to comment out `image = image[:, :, :3]`, which does 
                                nothing but convert from RGBA to RGB. 

        'annotation_data'; the data containing the annotations exported from labelstudio. Note that this
                            is more helpful if you merge `image data` and `annotation data` from your 
                            labelstudio exported json output.
        `newImageSize`: the new image size you wish to convert. It is a tuple and default is `(300, 300)`
    Returns:
            Returns a pandas dataframe with the new updated information on the newly resized and saved images.
    """
   

    newImageData = {
        'image_id' : [],
        'fileName': [],
        'segmentation': [],
        'bbox': [],
        'width': [],
        'height': []
    }

    # get all the image file names from the dataframe
    image_filenames = annotation_data['file_name'].tolist()

    # get the image filename from the dataframe
    for i in range(len(image_filenames)):
        # get file name by index location
        imageName = annotation_data.loc[i, 'file_name']
        # this comes in handy if we want the output as the 
        # previous dataframe but only with information we need.
        imageID = annotation_data.loc[i, 'image_id']
        imageWidth = newImageSize[0]
        imageHeight = newImageSize[1]

        # read image file fromm directory
        image = os.path.join(image_directory, imageName)
        # read image into numpy array
        image = imread(image)
        # convert image from `4` channels to `3 channels`
        image = image[:, :, :3]

        # get previous segmentation  and bounding box from the data
        segmentation, bbox = annotation_data.loc[i, 'segmentation'], annotation_data.loc[i, 'bbox']
        segmentation = np.array(segmentation).reshape((-1, 2))
        # get scaling 
        scale_x = newImageSize[0] / image.shape[1]
        scale_y = newImageSize[1] / image.shape[0]

        # resize
        image = cv2.resize(image, (newImageSize[0], newImageSize[1]))

        # get the bounding box
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        #1 trimming bbox
        xmin = int(np.round(float(xmin) * scale_x))
        ymin = int(np.round(float(ymin) * scale_y))
        xmax = int(np.round(float(xmax) * scale_x))
        ymax = int(np.round(float(ymax) * scale_y))
        bbox = [xmin, ymin, xmax, ymax]

        #2 update the segments
        segmentation[:, 0] *= scale_x
        segmentation[:, 1] *= scale_y
        segmentation = segmentation.reshape(1, -1).tolist()

        #3 save the new image to directory if not already exist. 
        fileName = os.path.join(target_image_directory,  imageName)
        if fileName not in os.listdir(target_image_directory):
            cv2.imwrite(fileName, image)
        # record new data
        newImageData['image_id'].append(imageID)
        newImageData['fileName'].append(imageName)
        newImageData['segmentation'].append(segmentation)
        newImageData['bbox'].append(bbox)
        newImageData['width'].append(imageWidth)
        newImageData['height'].append(imageHeight)
    
    return pd.DataFrame(newImageData)