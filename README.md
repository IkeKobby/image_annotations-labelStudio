# Image Segmentation
- Semantic segmentation with specific use case on buildings in Senegal. The project combined two data soources, hugely impacted by open source data from kaggle's data repository with use case on building segmentation. 
## Data setting 
- The images contained about 2k images with over 98% of the images from open source and remianing 2% from locally collected images from the satellite software, called the SASPLANET which works perfectly on a windows machine. 
- Over 200 images were collected with the software and annotated 100 images using using label-studio tool. The format of annotatation used is the `COCO` format. Due to some irregularities, the 100 annotated images were downsampled to select 50 best annotation to best fit with the open source dataset. This was after several unsuccessfull outcome with the entire 100 samples. 
## Data preprocessing - annotation resetting
A preprocessing function, `annotation_preproc` to help with the annotated images with label studio annotator for the `COCO` data format. Note that the `COCO` format comes with the following files if converted to pandas dataframe;

- 1:  The annotation files if converted to DF is shown below

| id | image_id |category_id | segmentation | bbox | iscrowd | ignore | area |
|----|----------|------------|---------------|-----|---------|--------|------|

- 2:  and the image data if converted to DF has the format

|width| height| id | file_name|  
|-----|-------|----|----------|

 - 3:  and most importantly and the required data is doing merge, left on `image_id`, right on `id` to have the following DF;

|id_x	|image_id	|category_id	|segmentation	|bbox	|ignore	|iscrowd	|area|	width	|height	|id_y	|file_name|
|-----|---------|-------------|-------------|-----|-------|---------|----|--------|-------|-----|---------|

- 4:  Let's say, i have `image_data_50` and `annotation_data_50` named `annotation_data_50` and `image_data_50`, do not mind the `50` it just means I annotated 50 images with `label-studio`, for more on label studio, [click here](https://labelstud.io/guide/). Here's a code for merging to get the `#3` DF.

 - `merge_data_50 = pd.merge(annotation_data_50, image_data_50, how='left', left_on = 'image_id', right_on='id').dropna()`

## Model training
- Several modeling experiments were conducted and finally resulted to the [keras-segmentation](https://github.com/divamgupta/image-segmentation-keras) model with different architectures.

### Training data;
- The model is designed to accept training images and training masks, both with the same `image-ids`. The data directory structure should be as follows; 
- ---- Training Data ----- 
- -----------------training images
- -----------------training masks

- ----Testing Data ---
- ----------------testing images
- ----------------testing masks.

The notebook `nb.ipynb` contains all the functions and methods used to generate corresponding masks for the images leveraging the `pycocotools` library with the help of the `coco` class method `annotMask`. 
- `NB`:  Both images and their corresponding masks must have the `name or id`. 

### Model usage
- After training the model is saved to a folder with different weights saved as hidden files, eg, of a saved weight is as of the form `.00020.data-00000-of-00001` with a config file in the form `_config.json`, a checkpoint file `checkpoint` and a forth file `.00020.index`. Also during training several of `.00020.data-00000-of-00001`, `.00020.index` are saved at each epoch and the model however, finds the last weight, this case, the last weight was on epoch `20`, hence the `.00020`.
- Loading a model requires all the above files in one directory and then using the `model_from_checkpoint` from the keras-segmentation module; `from keras_segmentation.predict import model_from_checkpoint_path`. The model can be loaded and instantiated for predictions. 
- Again the  model takes as input, an image and predict its mask. 