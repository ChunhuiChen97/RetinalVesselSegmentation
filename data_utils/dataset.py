import torch
from torch.utils.data import Dataset
class H5Dataset(Dataset):
     """Dataset wrapping data and target tensors.
 
     Each sample will be retrieved by indexing both tensors along the first
     dimension.
 
     Arguments:
         data_tensor (Tensor): contains sample data.
         target_tensor (Tensor): contains sample targets (labels).
     """
     def __init__(self, data_tensor, target_tensor):
         assert data_tensor.shape[0] == target_tensor.shape[0]
         self.data_tensor = data_tensor
         self.target_tensor = target_tensor
 
     def __getitem__(self, index):
         # print(index)
         return self.data_tensor[index], self.target_tensor[index]
 
     def __len__(self):
         return self.data_tensor.shape[0]

    