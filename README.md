# image_annotations-labelStudio
A preprocessing function to help with your annotated images with label studio annotator for the `COCO` data format. Note that the `COCO` format comes with the following files if converted to pandas dataframe;

- 1 The annotation files if converted to DF is shown below

| id | image_id |category_id | segmentation | bbox | iscrowd | ignore | area |
|----|----------|------------|---------------|-----|---------|--------|------|

- 2 and the image data if converted to DF has the format

|width| height| id | file_name|  
|-----|-------|----|----------|

 - 3 and most importantly and the required data is doing merge, left on `image_id`, right on `id` to have the following DF;

|id_x	|image_id	|category_id	|segmentation	|bbox	|ignore	|iscrowd	|area|	width	|height	|id_y	|file_name|
|-----|---------|-------------|-------------|-----|-------|---------|----|--------|-------|-----|---------|

- 4. Let's say, i have `image_data_50` and `annotation_data_50` named `annotation_data_50` and `image_data_50`, do not mind the `50` it just means I annotated 50 images with `label-studio`, for more on label studio, [click here](https://labelstud.io/guide/). Here's a code for merging to get the `#3` DF.

 - `merge_data_50 = pd.merge(annotation_data_50, image_data_50, how='left', left_on = 'image_id', right_on='id').dropna()`

