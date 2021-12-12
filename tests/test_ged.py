from graph_match.graph_match.models.SimGNN import SimGNN
from torch_geometric.datasets import GEDDataset
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree
import torch
import os.path as osp
import random
import numpy as np

random.seed(0)
np.random.seed(1)

DATASET_NAME = 'LINUX'
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Create node feature vectors that are one hot encoded by degree of each node
def create_node_vectors(train_graphs, test_graphs):
    if train_graphs[0].x is None:
        max_degree = 0
        for graph in train_graphs:
            # If this graph has edges then do
            if graph.edge_index.size(1) > 0:
                max_degree = max(max_degree, int(degree(graph.edge_index[0]).max().item()))
        
        # Create the feature matrix for the Dataset
        one_hot_degree = OneHotDegree(max_degree, cat=False)
        train_graphs.transform = one_hot_degree

    if test_graphs[0].x is None:
        max_degree = 0
        for graph in test_graphs:
            # If this graph has edges then do
            if graph.edge_index.size(1) > 0:
                max_degree = max(max_degree, int(degree(graph.edge_index[0]).max().item()))
        
        # Create the feature matrix for the Dataset
        one_hot_degree = OneHotDegree(max_degree, cat=False)
        test_graphs.transform = one_hot_degree

class PairedGEDDataset(Data):
    def __init__(self, edge_index_1, x1, edge_index_2, x2, ged, norm_ged, graph_sim):
        super(PairedGEDDataset, self).__init__()
        self.edge_index_1 = edge_index_1
        self.x1 = x1
        self.edge_index_2 = edge_index_2
        self.x2 = x2
        self.ged = ged
        self.norm_ged = norm_ged
        self.graph_sim = graph_sim

    def __inc__(self, key, value):
        if key == "edge_index_1":
            return self.x1.size(0)
        elif key == "edge_index_2":
            return self.x2.size(0)
        else:
            return super().__inc__(key, value)

def evaluate_performance(loader, model, loss_criterion):
    total_loss = 0
    num_ex = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            sim_pred = model(data)
            batch_loss = loss_criterion(sim_pred, data.graph_sim)
            total_loss = total_loss + batch_loss*len(data.ged)
            num_ex += len(data.ged)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    return total_loss.item()/num_ex

if __name__ == "__main__":
    path = osp.join('data/simgnn', DATASET_NAME)
    train_graphs = GEDDataset(root = path, train = True, name = DATASET_NAME)
    test_graphs = GEDDataset(root = path, train = False, name = DATASET_NAME)
    
    create_node_vectors(train_graphs, test_graphs)
    num_node_features = train_graphs.num_features
    num_edge_features = train_graphs.num_edge_features

    # Data List to pass into the Data Loader to get Batches
    train_graph_pair_list = []
    test_graph_pair_list = []

    # Making the Pairs of Graphs
    for graph1_idx, graph1 in enumerate(train_graphs):
        for graph2 in train_graphs:
            # Initializing Data
            edge_index_1 = graph1.edge_index
            x1 = graph1.x
            edge_index_2 = graph2.edge_index
            x2 = graph2.x
            ged = train_graphs.ged[graph1.i, graph2.i]
            norm_ged = train_graphs.norm_ged[graph1.i, graph2.i]
            graph_sim = torch.exp(-norm_ged)
            
            # Making Graph Pair
            graph_pair = PairedGEDDataset(edge_index_1=edge_index_1, x1=x1, 
                                        edge_index_2=edge_index_2, x2=x2,
                                        ged=ged ,norm_ged=norm_ged, graph_sim = graph_sim)
            
            # Saving all the Graph Pairs to the List for Batching and Data Loading
            train_graph_pair_list.append(graph_pair)

    for graph1_idx, graph1 in enumerate(test_graphs):
        for graph2 in train_graphs:
            # Initializing Data
            edge_index_1 = graph1.edge_index
            x1 = graph1.x
            edge_index_2 = graph2.edge_index
            x2 = graph2.x
            ged = train_graphs.ged[graph1.i, graph2.i]
            norm_ged = train_graphs.norm_ged[graph1.i, graph2.i]
            graph_sim = torch.exp(-norm_ged)
            
            # Making Graph Pair
            graph_pair = PairedGEDDataset(edge_index_1=edge_index_1, x1=x1, 
                                        edge_index_2=edge_index_2, x2=x2,
                                        ged=ged ,norm_ged=norm_ged, graph_sim = graph_sim)
            
            # Saving all the Graph Pairs to the List for Batching and Data Loading
            test_graph_pair_list.append(graph_pair)

    val_graph_pair_list = random.sample(train_graph_pair_list, len(test_graph_pair_list))
    train_graph_pair_list = list(set(train_graph_pair_list) - set(val_graph_pair_list))

    print("Number of Train Pairs = {}".format(len(train_graph_pair_list)))
    print("Number of Validation Pairs = {}".format(len(val_graph_pair_list)))
    print("Number of Test Pairs = {}".format(len(test_graph_pair_list)))

    train_loader = DataLoader(train_graph_pair_list, batch_size = TRAIN_BATCH_SIZE, follow_batch = ["x1", "x2"], shuffle = True)
    val_loader = DataLoader(val_graph_pair_list, batch_size = TRAIN_BATCH_SIZE, follow_batch = ["x1", "x2"], shuffle = True)
    test_loader = DataLoader(test_graph_pair_list, batch_size = TEST_BATCH_SIZE, follow_batch = ['x1', 'x2'], shuffle = True)

    # Start Training
    model = SimGNN(num_node_features)
    loss_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 7
    train_loss_arr = []
    val_loss_arr = []

    for epoch in range(epochs):
        model.train()
        for batch_idx, (train_batch, val_batch) in enumerate(zip(train_loader, val_loader)):
            train_batch = train_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(train_batch)
            loss = loss_criterion(y_pred, train_batch.graph_sim)
            # Compute Gradients via Backpropagation
            loss.backward()
            # Update Parameters
            optimizer.step()
            train_loss_arr.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_batch = val_batch.to(device)
            y_val_pred = model(val_batch)
            val_loss = loss_criterion(y_val_pred, val_batch.graph_sim)
            val_loss_arr.append(val_loss.item())

        # Printing Loss Values
        if batch_idx%200 == 0:
            print(f"Epoch{epoch+1}/{epochs} | Batch: {batch_idx} | Train Loss: {loss} | Validation Loss: {val_loss}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
    
    # Printing Epoch Summary
    print(f"Epoch: {epoch+1}/{epochs} | Train MSE: {loss} | Validation MSE: {val_loss}")

