
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

def get_mask(coco, img_id, cat_ids, annot_df):
    
    img_shape = (300, 300, 1)

    ann_ids = annot_df[annot_df['image_id'] == img_id].copy().reset_index(drop = True)

    anns = [ann_ids.loc[i] for i in range(len(ann_ids))]


    masks = np.zeros(img_shape)
    for idx, cat_id in enumerate(cat_ids):
        mask = np.zeros(img_shape[:2])
        for ann in anns:
            if cat_id == ann['category_id']:
                mask = np.maximum(mask, coco.annToMask(ann))
        masks[:, :, idx] = mask
    return masks


## data augmentation
def data_augmentation_training(image, mask, width=256, height=256):
    transform = A.Compose(
                        [ 
                            A.Rotate(limit=45, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.1),
                            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
                            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                            A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
                            A.RandomCrop(height=height, width=width, always_apply=True),
                            A.IAAAdditiveGaussianNoise(p=0.2),
                            A.IAAPerspective(p=0.5),
                            A.OneOf([
                                    A.Blur(blur_limit=3, p=0.5),
                                ], p=1.0),
                            A.OneOf([
                                    A.CLAHE(p=1),
                                    A.RandomBrightness(p=1),
                                    A.RandomGamma(p=1)
                                ], p=1.0),
                            A.OneOf([
                                    A.IAASharpen(p=1),
                                    A.Blur(blur_limit=3, p=1),
                                    A.MotionBlur(blur_limit=3, p=1),
                                ],p=0.9,),
                            A.OneOf([
                                    A.RandomContrast(p=1),
                                    A.HueSaturationValue(p=1),
                                    ],p=0.9),
                            A.Normalize(
                                mean= [0,0,0],
                                std=[1,1,1],
                                max_pixel_value=255
                            ),
                            ToTensorV2(),
                        ]
                    )
    transformed = transform(image = image, mask = mask)
    image = transformed['image']
    mask = transformed['mask']
    return image,mask

## validation transform
def data_augmentation_validation(image, mask, width=256, height=256):
    transform = A.Compose(
        [
            A.Resize(width=width, height=height),
            ToTensorV2(),
        ]
    )
    transformed = transform(image = image, mask = mask)
    image = transformed['image']
    mask = transformed['mask']
    return image.float(), mask



# Preprocess
def cropping(input_image, scale_percentage = .50, default = True):
    if default:
        dim = (324, 324)
    else:
        width = int(input_image.shape[1] * scale_percentage)
        height = int(input_image.shape[0] * scale_percentage)
        dim = (width, height)
    return cv2.resize(input_image, dim,)


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
