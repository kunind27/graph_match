import torch
from torch_geometric.nn.conv import GCNConv, SAGEConv
from torch_geometric.data import Data, Batch
from utils.utility import cudavar

# TODO: 
# 1. Revise bottleneck business, K and hist length are both 16 hence more layers needed
# 2. Can user be flexible with number of GCN layers for desirable feature extraction? 
# 3. Ask Indra about summation technique used for passing features to attention layer weights
# 4. Start including flags for attention mechanisms and encoder techniques
# 5. Should GNN have different activations? (ReLU is in the paper)

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
        self.attention_layer = torch.nn.Linear(self.gcn_filter_list[2], self.gcn_filter_list[2], bias = False)
        torch.nn.init.xavier_uniform_(self.attention_layer.weight)

        # NTN capturing graph-graph interaction
        # Output is R^k vector at different scales k (tensor_neurons)
        self.ntn_w = torch.nn.Bilinear(self.gcn_filter_list[2], self.gcn_filter_list[2], self.tensor_neurons, bias = False)
        torch.nn.init.xavier_uniform_(self.ntn_w.weight)
        self.ntn_v = torch.nn.Linear(2 * self.gcn_filter_list[2], self.tensor_neurons, bias = False)
        torch.nn.init.xavier_uniform_(self.ntn_v.weight)
        self.ntn_bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))
        torch.nn.init.xavier_uniform_(self.ntn_bias.weight)

        # Feature Count for histogram business
        feature_count = (self.tensor_neurons + self.bins) if self.bins else self.tensor_neurons
        # for now only one bottle neck layer is implemented (therefore FCN has only one hidden layer)
        self.fc1 = torch.nn.Linear(feature_count, self.bottle_neck_neurons)
        self.fc2 = torch.nn.Linear(self.bottle_neck_neurons, 1) 
    
    def GNN(self, data, dropout: float = 0):
        features = self.conv1(data.x, data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p = dropout, training = self.training)

        features = self.conv2(features,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p = dropout, training = self.training)

        features = self.conv3(features, data.edge_index)

        return features

    def forward(self, batch_data, batch_data_sizes):
        q_graphs, c_graphs = zip(*batch_data)
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(torch.tensor(a))
        cgraph_sizes = cudavar(torch.tensor(b))
        query_batch = Batch.from_data_list(q_graphs)
        query_batch.x = self.GNN(query_batch)
        query_gnode_embeds = [g.x for g in query_batch.to_data_list()]

        
