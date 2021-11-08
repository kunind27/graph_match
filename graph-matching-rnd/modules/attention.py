import torch
from torch.nn.utils.rnn import pad_sequence
from utils.utility import cudavar

class AttentionLayer(torch.nn.Module):
    def __init__(self, input_dim, type: str = 'simgnn', activation: str = "tanh", a = 0.1):
        """
        :param: type: Type of attention mechanism to be used
        :param: input_dim: Input Dimension of the Node Embeddings
        :param: activation: The Activation Function to be used for the Attention Layer
        :param: a: Slope of the -ve part if the activation is Leaky ReLU
        """
        super(AttentionLayer, self).__init__()
        self.type = type
        self.d = input_dim # Input dimension of the node embeddings
        self.activation = activation 
        self.a = a # Slope of the negative part in Leaky-ReLU
        
        self.params()
        self.initialize()
        
    def params(self):
        if(self.type == 'simgnn'):
            self.W_att = torch.nn.Parameter(torch.Tensor(self.d, self.d))

    def initialize(self):
        """
        Weight initialization depends upon the activation function used.
        If ReLU/ Leaky ReLU : He (Kaiming) Initialization
        If tanh/ sigmoid : Xavier Initialization
        """
        if self.activation == "leaky_relu" or self.activation == "relu":
            torch.nn.init.kaiming_normal_(self.W_att, a = self.a, nonlinearity = self.activation)
        elif self.activation == "tanh" or self.activation == "sigmoid":
            torch.nn.init.xavier_normal_(self.W_att)
        else:
            raise ValueError("Activation can only take values: 'relu', 'leaky_relu', 'sigmoid', 'tanh';\
                            {} is invalid".format(self.activation))

    def forward(self, graph_batch, graph_sizes):
        """ 
        :param: graph_batch : Batch containing graphs
        :param: size : Check Documentation https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/mean.html
        :return: global_graph_embedding for each graph in the batch
        """       
        node_embeds = [g.x for g in graph_batch]
        q = pad_sequence(node_embeds, batch_first=True)
        graph_sizes = cudavar(torch.tensor(graph_sizes))
        context = torch.div(torch.matmul(self.W_att, torch.sum(q, dim=1).T), graph_sizes).T
        
        activations = {"tanh": torch.nn.functional.tanh, "leaky_relu": torch.nn.functional.leaky_relu,
                        "relu": torch.nn.functional.relu, "sigmoid": torch.nn.functional.sigmoid}
        _activation = activations[self.activation]
        # Applying the Non-Linearity over W_att*mean(U_i), the default is tanh
        context = _activation(context)

        sigmoid_scores = torch.sigmoid(q@context.unsqueeze(2))
        e = (q.permute(0,2,1)@sigmoid_scores).squeeze() 
        
        return e