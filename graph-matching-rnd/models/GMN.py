import torch
from torch_geometric.nn.conv import GCNConv, SAGEConv
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence
from utils.utility import cudavar