import torch
from torch_geometric.nn.conv import GCNConv, SAGEConv
from torch.nn import Linear, Bilinear, Parameter

# TODO: 
# 1. Revise bottleneck business, K and hist length are both 16 hence more layers needed
# 2. Can user be flexible with number of GCN layers for desirable feature extraction? 
# 3. Ask Indra about summation technique used for passing features to attention layer weights
# 4. Start including flags for attention mechanisms and encoder techniques

class SimGNN(torch.nn.module):
    def __init__(self, input_dim: int, tensor_neurons: int = 16, filters: list = [64, 32, 16],
                 bottle_neck: int = 16, hist_bins: int = 0):
        super(SimGNN, self).__init__()
        self.input_dim = input_dim
        self.gcn_filter_list = filters
        self.tensor_neurons = tensor_neurons
        self.bins = hist_bins
        self.bottle_neck_neurons = bottle_neck

        # Initialise with 3 GCN layers
        if(len(self.gcn_filter_list) == 3):
            self.conv1 = GCNConv(self.input_dim, self.gcn_filter_list[0])
            self.conv2 = GCNConv(self.gcn_filter_list[0], self.gcn_filter_list[1])
            self.conv3 = GCNConv(self.gcn_filter_list[1], self.gcn_filter_list[2])
        else:
            raise RuntimeError(
                f"Number of GCN layers "
                f"'{len(self.gcn_filter_list)}' should be 3")

        # Attention layer mechanism that operates over sum of node embeddings
        self.attention_layer = Linear(self.gcn_filter_list[2], self.gcn_filter_list[2], bias = False)
        torch.nn.init.xavier_uniform_(self.attention_layer.weight)

        # NTN capturing graph-graph interaction
        # Output is R^k vector at different scales k (tensor_neurons)
        self.ntn_w = Bilinear(self.gcn_filter_list[2], self.gcn_filter_list[2], self.tensor_neurons, bias = False)
        torch.nn.init.xavier_uniform_(self.ntn_w.weight)
        self.ntn_v = Linear(2 * self.gcn_filter_list[2], self.tensor_neurons, bias = False)
        torch.nn.init.xavier_uniform_(self.ntn_v.weight)
        self.ntn_bias = Parameter(torch.Tensor(self.tensor_neurons, 1))
        torch.nn.init.xavier_uniform_(self.ntn_bias.weight)

        # Feature Count for histogram business
        feature_count = (self.tensor_neurons + self.bins) if self.bins else self.tensor_neurons
        # for now only one bottle neck layer is implemented (therefore FCN has only one hidden layer)
        self.fc1 = Linear(feature_count, self.bottle_neck_neurons)
        self.fc2 = Linear(self.bottle_neck_neurons, 1) 




