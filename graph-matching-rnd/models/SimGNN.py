import torch
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional

from modules.attention import AttentionLayer
from modules.encoder import GraphEncoder

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
                 bottle_neck: int = 16, hist_bins: int = 0, conv: str = "gcn", activation = "tanh"):
        super(SimGNN, self).__init__()
        self.input_dim = input_dim
        self.conv_filter_list = filters
        self.activation = activation
        self.setHyperParams(tensor_neurons, hist_bins)
        self.bottle_neck_neurons = bottle_neck
        self.conv_type = conv

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

    def setupLayers(self):
        self.conv_layer = GraphEncoder(self.input_dim, None, filters = self.conv_filter_list,
                                        conv_type = self.conv_type, name = "simgnn")
        self.attention_layer = AttentionLayer(self.input_dim, type = 'simgnn', activation = self.activation)

    def GNN(self, data, dropout: float = 0):
        features = self.conv1(data.x, data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p = dropout, training = self.training)

        features = self.conv2(features,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p = dropout, training = self.training)

        features = self.conv3(features, data.edge_index)

        return features

    def forward(self, batch_data, batch_data_sizes, conv_dropout: int = 0, isolate = None):
        """
        Forward pass with query and corpus graphs.
        :param data: A Batch Containing a Pair of Graphs.
        :return score: Similarity score.
        """
        q_graphs, c_graphs = zip(*batch_data)
        a,b = zip(*batch_data_sizes)
        query_batch = Batch.from_data_list(q_graphs)
        corpus_batch = Batch.from_data_list(c_graphs)
        
        query_batch.x = self.conv_layer(query_batch.x, query_batch.edge_index, dropout = conv_dropout)
        corpus_batch.x = self.conv_layer(corpus_batch.x, corpus_batch.edge_index, dropout = conv_dropout)

        query_g_emb = self.attention_layer(query_batch.to_data_list(), a)
        corpus_g_emb = self.attention_layer(corpus_batch.to_data_list(), b)

        if isolate == "att":
            return query_g_emb, corpus_g_emb
        elif isolate is not None:
            raise ValueError("Invalid value of argument:", isolate)
        
        # Pass attention based graph embeddings to NTN and obtain similarity scores
        scores = torch.nn.functional.relu(self.ntn_a(query_g_emb, corpus_g_emb) + 
                                        self.ntn_b(torch.cat((query_g_emb, corpus_g_emb),dim=-1)) + self.ntn_bias.squeeze())

        # Concatenate histogram of pairwise node-node interaction scores if specified
        if self.bins:
            query_node_emb, corpus_node_emb = pad_sequence([g.x for g in query_batch.to_data_list()], batch_first = True), \
                                                pad_sequence([c.x for c in corpus_batch.to_data_list()], batch_first = True)
            h = torch.histc(query_node_emb@corpus_node_emb.permute(0,2,1),bins=self.bins)
            h = torch.div(h, torch.sum(h))

            scores = torch.cat((scores, h), dim = 1)

        scores = torch.nn.functional.relu(self.fc1(scores))
        score = torch.sigmoid(self.fc2(scores))
        preds = []
        preds.append(score)
        p = torch.stack(preds).squeeze()
        
        return p