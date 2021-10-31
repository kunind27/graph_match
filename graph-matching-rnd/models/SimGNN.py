import torch
from torch_geometric.nn.conv import GCNConv, SAGEConv
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence
from utils.utility import cudavar

# TODO: 

# High Priority
# 1. Revise bottleneck business, K and hist length are both 16 hence more layers needed
# 2. Can user be flexible with number of Conv layers for desirable feature extraction? 
# 3. Ask Indra about summation technique used for passing features to attention layer weights
# 4. Start including flags for attention mechanisms and encoder techniques
# 5. Should GNN have different activations? (ReLU is in the paper)
# 6. Include GIN and GAT Conv mechanisms
# 7. Figure out if node encoding can be done internally (will let us use different kinds of
#    encoding mechanisms like one-hot, adj-list ???)
# 8. Does isolating an attention mechanism require returning for both query and corpus graphs?

# Low Priority
# 6. Figure out how different Conv mechanisms work, assumed same for now.

class SimGNN(torch.nn.module):
    def __init__(self, input_dim: int, tensor_neurons: int = 16, filters: list = [64, 32, 16],
                 bottle_neck: int = 16, hist_bins: int = 0, conv: str = "gcn"):
        super(SimGNN, self).__init__()
        self.input_dim = input_dim
        self.conv_filter_list = filters
        self.setHyperParams(tensor_neurons, hist_bins)
        self.bottle_neck_neurons = bottle_neck
        self.conv_type = conv
        
        # Initialise with 3 GCN layers
        if(len(self.conv_filter_list) == 3):
            if(self.conv_type == "gcn"):
                self.conv1 = GCNConv(self.input_dim, self.conv_filter_list[0])
                self.conv2 = GCNConv(self.conv_filter_list[0], self.conv_filter_list[1])
                self.conv3 = GCNConv(self.conv_filter_list[1], self.conv_filter_list[2])
            elif(self.conv_type == "sage"):
                self.conv1 = SAGEConv(self.input_dim, self.conv_filter_list[0])
                self.conv2 = SAGEConv(self.conv_filter_list[0], self.conv_filter_list[1])
                self.conv3 = SAGEConv(self.conv_filter_list[1], self.conv_filter_list[2])
        else:
            raise RuntimeError(
                f"Number of Convolutional layers "
                f"'{len(self.conv_filter_list)}' should be 3")

        # Attention layer mechanism that operates over sum of node embeddings
        self.attention_layer = torch.nn.Linear(self.conv_filter_list[2], self.conv_filter_list[2], bias = False)
        torch.nn.init.xavier_uniform_(self.attention_layer.weight)

        # NTN capturing graph-graph interaction
        # Output is R^k vector at different scales k (tensor_neurons)
        self.ntn_w = torch.nn.Bilinear(self.conv_filter_list[2], self.conv_filter_list[2], self.tensor_neurons, bias = False)
        torch.nn.init.xavier_uniform_(self.ntn_w.weight)
        self.ntn_v = torch.nn.Linear(2 * self.conv_filter_list[2], self.tensor_neurons, bias = False)
        torch.nn.init.xavier_uniform_(self.ntn_v.weight)
        self.ntn_bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))
        torch.nn.init.xavier_uniform_(self.ntn_bias.weight)

        # Feature Count for histogram business
        feature_count = (self.tensor_neurons + self.bins) if self.bins else self.tensor_neurons
        # for now only one bottle neck layer is implemented (therefore FCN has only one hidden layer)
        self.fc1 = torch.nn.Linear(feature_count, self.bottle_neck_neurons)
        self.fc2 = torch.nn.Linear(self.bottle_neck_neurons, 1) 
    
    def setHyperParams(self, k: int, bins: int):
        # Output Dimension of the NTN
        self.tensor_neurons = k
        # No. of Bins to be used for the Histogram 
        self.bins = bins

    def GNN(self, data, dropout: float = 0):
        features = self.conv1(data.x, data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p = dropout, training = self.training)

        features = self.conv2(features,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p = dropout, training = self.training)

        features = self.conv3(features, data.edge_index)

        return features

    def forward(self, batch_data, batch_data_sizes, isolate = None):
        # Entire section's data handling needs to be reworked
        q_graphs, c_graphs = zip(*batch_data)
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(torch.tensor(a))
        cgraph_sizes = cudavar(torch.tensor(b))
        query_batch = Batch.from_data_list(q_graphs)
        query_batch.x = self.GNN(query_batch)
        # Query Graph Node Embeddings
        query_gnode_embeds = [g.x for g in query_batch.to_data_list()]

        corpus_batch = Batch.from_data_list(c_graphs)
        corpus_batch.x = self.GNN(corpus_batch)
        # Corpus Graph Node Embeddings
        corpus_gnode_embeds = [g.x for g in corpus_batch.to_data_list()]

        # Obtain sigmoid attention weights and aggregate their product with node embeddings
        q = pad_sequence(query_gnode_embeds,batch_first=True)
        context = torch.tanh(torch.div(torch.sum(self.attention_layer(q),dim=1).T,qgraph_sizes).T)
        sigmoid_scores = torch.sigmoid(q@context.unsqueeze(2))
        e1 = (q.permute(0,2,1)@sigmoid_scores).squeeze()
        
        c = pad_sequence(corpus_gnode_embeds,batch_first=True)
        context = torch.tanh(torch.div(torch.sum(self.attention_layer(c),dim=1).T,cgraph_sizes).T)
        sigmoid_scores = torch.sigmoid(c@context.unsqueeze(2))
        e2 = (c.permute(0,2,1)@sigmoid_scores).squeeze()

        if isolate == "att":
            return e1, e2
        elif isolate is not None:
            raise ValueError("Invalid value of argument:", isolate)
        
        # Pass attention based graph embeddings to NTN and obtain similarity scores
        scores = torch.nn.functional.relu(self.ntn_a(e1,e2) +self.ntn_b(torch.cat((e1,e2),dim=-1))+self.ntn_bias.squeeze())

        # Concatenate histogram of pairwise node-node interaction scores if specified
        if self.bins:
            h = torch.histc(q@c.permute(0,2,1),bins=self.bins)
            h = torch.div(h, torch.sum(h))

            scores = torch.cat((scores, h), dim = 1)

        scores = torch.nn.functional.relu(self.fc1(scores))
        score = torch.sigmoid(self.fc2(scores))
        preds = []
        preds.append(score)
        p = torch.stack(preds).squeeze()
        
        return p