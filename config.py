from .domains.overnight.OvernightDataset import OvernightDataset, train_dict, dev_dict, test_all
from .models.sdpe_model import SDPE
import torch
import os


PATH = os.path.dirname(os.path.abspath(__file__))
max_len = 64
diff_step = 250
initializing = PATH + '/base/bert-tiny'#'base/bert-mini'
checkpoint = PATH + 'domains/overnight/checkpoints/sdpe-250/checkpoint-2000'
device = torch.device('cuda')
model = SDPE(initializing,max_len,diff_step)
state = torch.load(initializing+'/pytorch_model.bin', map_location=device) #"/Saved_Models/20220903bert_diffusion/bestloss.pkl")

model_dict = model.state_dict()
# 1. filter out unnecessary keys
if list(state.keys())[0].startswith("module."):
    state = {k[7:]: v for k, v in state.items() if k[7:] in model_dict}
else:
    state = {k: v for k, v in state.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(state)
# 3. load the new state dict
model.load_state_dict(model_dict)

# model.load_state_dict(state,strict=True)
model = model.to(device)
model.eval()
print("Trial 31")

train_set = OvernightDataset(train_dict, init_model=initializing, max_len=max_len)
val_set = OvernightDataset(dev_dict, init_model=initializing, max_len=max_len)
test_set = OvernightDataset(test_all, init_model=initializing, max_len=max_len)
