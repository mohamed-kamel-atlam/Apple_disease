import torch

# Specifies the appropriate device to be used (either GPU or CPU) to perform the operations
def get_default_device () :
    if torch.cuda.is_available () :            # Check GPU availability
        return torch. device ('cuda' )  
    else:                                      # If GPU is not available
        return torch.device ('cpu' )
# Transfers data to the specified device (GPU or CPU)    
def to_device (data, device) :
    if isinstance (data, (list, tuple) ) :    # Dealing with complex data (List or Tuple)
        return [to_device (x, device) for x in data]   # Transfer every item inside it to the device
    return data.to (device, non_blocking=True)   # Transfer Tensor to device

# To automatically move data to the appropriate device (such as GPU or CPU) during batch iterations
class DeviceDataLoader () :
    # Constructor
    def __init__(self, dl, device) :
        self.dl = dl            # DataLoader
        self.device = device    # ('cuda' OR 'cpu')
    
    # Makes the class repeatable
    def __iter__(self) :
        for b in self.dl:
            '''Batch returns without terminating the iteration,
            which makes it suitable for handling large amounts of data efficiently'''
            yield to_device (b, self.device) # Transfer the batch to the device
            
    # Returns the number of batches        
    def __len__(self) :
        return len (self.dl)