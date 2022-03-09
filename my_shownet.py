from torch.utils.tensorboard import SummaryWriter
from Code.model_lung_infection.InfNet_VGGNet import Inf_Net
import torch

writer = SummaryWriter("./logs")
model = Inf_Net(channel=32, n_class=1)
input = torch.randn(1, 3, 352, 352)
writer.add_graph(model,input)
writer.close()

