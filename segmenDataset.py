import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from data.utils import get_mask


class segmentationDataset(Dataset):
    """
    Customed dataset class for image segmentation on buildings
    in Senegal.
    """
    def __init__(self, root_dir, 
                        img_ids, 
                        cat_ids, 
                        coco_api, 
                        img_df,
                        annot_df,
                        transforms=None, 
                        preprocessing=None):
        
        self.root_dir = root_dir
        self.img_ids = img_ids
        self.cat_ids = cat_ids
        self.coco_api = coco_api
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.img_df = img_df
        self.annot_df = annot_df
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx] # get image idx
        img_inf = self.img_df[img_id] # load image info
        
        file_name = img_inf['file_name'] # get the image file name
        file_path = f'{self.root_dir}/{file_name}'
        img = Image.open(file_path).convert('RGB')
        mask = get_mask(self.coco_api, img_id, self.cat_ids, self.annot_df)

        if self.transforms is not None:
            return self.transforms(np.array(img), mask)
        return np.array(img), mask
    
    def __len__(self):
        return len(self.img_ids)

