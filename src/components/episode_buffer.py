from typing import Dict
import torch
from torch import Tensor

class EpisodeBatch:
    """Simple episode data storage."""
    def __init__(self, scheme, groups, batch_size, max_seq_length, device):
        self.scheme = scheme
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device
        
        self.data = {}
        self._setup_data()
        
    def _setup_data(self):
        """Initialize data storage structures."""
        for field_key, field_info in self.scheme.items():
            vshape = field_info["vshape"]
            dtype = field_info.get("dtype", torch.float32)
            
            if isinstance(vshape, int):
                vshape = (vshape,)
                
            if "group" in field_info:
                shape = (self.batch_size, self.max_seq_length, 
                        self.groups[field_info["group"]]) + vshape
            else:
                shape = (self.batch_size, self.max_seq_length) + vshape
                
            self.data[field_key] = torch.zeros(
                shape, dtype=dtype, device=self.device
            )
            
    def update(self, data, ts):
        """Update data at timestep."""
        for k, v in data.items():
            if k in self.data:
             
                if k in self.scheme and "group" in self.scheme[k]:
                   
                    if isinstance(v, list):
                       
                        if all(isinstance(item, torch.Tensor) for item in v):
                           
                            v = torch.stack(v, dim=0)  # (n_agents, ...)
                        else:
                          
                            v = torch.tensor(v, device=self.device)
                    
              
                    if isinstance(v, torch.Tensor):
                        v = v.to(self.device)
                   
                        expected_agents = self.groups[self.scheme[k]["group"]]
                        if v.shape[0] != expected_agents:
                            raise ValueError(f"Expected {expected_agents} agents for field '{k}', got {v.shape[0]}")
                        self.data[k][:, ts] = v
                    else:
                        raise TypeError(f"Expected tensor or list of tensors for grouped field '{k}', got {type(v)}")
                else:

                    if isinstance(v, list):

                        if len(v) == 1:
                            v = v[0]
                        else:
                            v = torch.tensor(v, device=self.device)
                    
                    if isinstance(v, torch.Tensor):
                        v = v.to(self.device)
                    
                    self.data[k][:, ts] = v
                
    def __getitem__(self, item):
        """Get data by key or slice."""
        if isinstance(item, str):

            return self.data[item]
        elif isinstance(item, slice):

            sliced_batch = EpisodeBatch(
                self.scheme, self.groups, 
                self.batch_size, self.max_seq_length, self.device
            )

            for key, tensor in self.data.items():
                sliced_batch.data[key] = tensor[item]
            return sliced_batch
        else:

            indexed_batch = EpisodeBatch(
                self.scheme, self.groups, 
                1 if isinstance(item, int) else self.batch_size, 
                self.max_seq_length, self.device
            )
            for key, tensor in self.data.items():
                indexed_batch.data[key] = tensor[item]
            return indexed_batch