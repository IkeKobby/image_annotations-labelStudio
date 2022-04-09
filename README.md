# image_annotations-labelStudio
A preprocessing function to help with your annotated images with label studio annotator for the `COCO` data format. Note that the `COCO` format comes with the following files if converted to pandas dataframe; 

|----|----------|------------|---------------|-----|---------|--------|------|
| id | image_id |category_id | segmentation | bbox | iscrowd | ignore | area |
|----|----------|------------|---------------|-----|---------|--------|------|
