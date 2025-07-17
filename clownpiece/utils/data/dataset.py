from typing import Callable, List, Any, Union
import os
from PIL import Image
import numpy as np

from clownpiece.tensor import Tensor

class Dataset:
  
  def __init__(self):
    pass

  def __getitem__(self, index):
    """
    Returns the item at the given index.
    """
    raise NotImplementedError("Dataset __getitem__ method not implemented")
  
  def __len__(self):
    """
    Returns the total number of item
    """
    raise NotImplementedError("Dataset __len__ method not implemented")
  
"""
CSV
"""
  
class CSVDataset(Dataset):

    file_path: str
    data: List[Any]
    transform: Callable

    def __init__(self, file_path: str, transform: Callable = None):
        # load CSV, apply transform
        self.file_path = file_path
        self.transform = transform
        self.data = []
        self.load_data()

    def load_data(self):
        # read CSV and store transformed rows
        # should be called at the end of __init__
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            self.data = [line.strip().split(',') for line in lines]
            print(self.data)
        
        if self.transform:
            self.data = [self.transform(row) for row in self.data]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

"""
Image
"""

class ImageDataset(Dataset):

    file_path: str
    data: List[Union[np.ndarray, Tensor]]
    labels: List[int]
    transform: Callable
    class_to_idx: dict[str, int]

    def __init__(self, file_path: str, transform: Callable = None):
        self.file_path = file_path
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        self.load_data()

    def load_data(self):
        # 1. read the subdirectories
        classes = sorted(entry.name for entry in os.scandir(self.file_path) if entry.is_dir())
        
        # 2. assign label_id for each subdirectory (i.e., class label)
        for i, cls_name in enumerate(classes):
            self.class_to_idx[cls_name] = i
            cls_path = os.path.join(self.file_path, cls_name)
            
            # 3. read files in subdirectory
            files = sorted(entry.name for entry in os.scandir(cls_path) if entry.is_file())
            
            for file_name in files:
                # 4. convert PIL Image to np.ndarray
                file_path = os.path.join(cls_path, file_name)
                try:
                    img = Image.open(file_path).convert('RGB')
                    img_array = np.array(img)
                
                    # 5. apply transform
                    if self.transform:
                        img_array = self.transform(img_array)
                    
                    # 6. store transformed image and label_id
                    self.data.append(img_array)
                    self.labels.append(i)
                
                except Exception as e:
                    print(f"Skipping {cls_path} due to error: {e}")

    def __getitem__(self, index):
        # index->(image, label_id)
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
  
"""
Image Transforms
"""

# These are functions that return desired transforms
#   args -> (np.ndarray -> np.ndarray or Tensor)
def sequential_transform(*trans):
    def compose(img: np.ndarray):
        for transform in trans:
            img = transform(img)
        return img
    return compose

def resize_transform(size):
    if not isinstance(size, tuple):
        size = (size, size)
    def resize(img: np.ndarray):
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize(size, Image.BILINEAR)
        return np.array(img_pil)
    return resize        

def normalize_transform(mean, std):
    def normalize(img: np.ndarray):
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        return img
    return normalize

def to_tensor_transform():
    def to_tensor(img: np.ndarray):
        return Tensor(img.tolist())
    return to_tensor