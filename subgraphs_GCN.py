from itertools import product
from torch.optim import Adam, SGD
from model_runner import main_gcn
from torch_geometric.datasets import Planetoid, CoraFull, Coauthor, PPI
import os
import torch
import numpy as np
import networkx as nx
import pickle




#dataSetName = "cora"; num_classes = 7; avarage_deg  = 3.8980797636632203 #tested with 0.5 7? test and 8 trials
dataSetName = "CiteSeer"; num_classes = 6; avarage_deg  = 2.7363991584009617 #tested with 0.5 test and 8 trials
#dataSetName = "PubMed"; num_classes = 3; avarage_deg  = 4.496018664096972 #test with 0.4 only, 3 trials only
#cora 50 50 trials. citeseer 60 epochs 50 trials. PubMed 50 for 10 trials with 0.5 train (usual) and 0.05 test (rest were 0.1)


#notes: gnx matches the planetoid - checked l

class GCN_subgraphs:
    def __init__(self, nni=False):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._nni = nni

    def _load_data(self):
        with open(os.path.join("dataSets","ego_graphs_"+dataSetName+".pkl"), 'rb') as f:
            graphs = pickle.load(f)

        'load using torch_geometric datasets (Roee)'
        data_transform = None
        print("loading data")
        self._data_path = './DataSets/{}'.format(dataSetName)
        if dataSetName == "CoraFull":
            self._data_set = CoraFull(self._data_path)
        elif dataSetName in {"CS", "Physics"}:
            self._data_set = Coauthor(self._data_path, dataSetName)
        else:
            self._data_set = Planetoid(self._data_path, dataSetName, data_transform)

        self._data_set.data.to(self._device)
        self._data = self._data_set[0]
        labels = self._data.y
        self._X = self._data.x





        'load using nx graphs'
        # with open(os.path.join("dataSets", "labels_"+dataSetName+".pkl"), 'rb') as f:
        #     labels = pickle.load(f)

        # #  move labels to cpu todo make on all graphs
        # labels=labels.to('cpu')
        # with open(os.path.join("dataSets", "labels_" + dataSetName + ".pkl"), 'wb') as handle:
        #   pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # was removed so we load it each time for each train ...
        # with open(os.path.join("dataSets", "X_manipulations_"+dataSetName+".pkl"), 'rb') as f:
        #     X = pickle.load(f)

        #self._feature_matrices = mx_s todo
        # self._X = X

        #_adjacency_matrices is for every node, since ego graph is for evey node
        self._adjacency_matrices = graphs #[nx.adjacency_matrix(g).tocoo() for g in graphs]
        self._labels = labels

    def train(self, input_params=None):
        if input_params is None:

            #  lr 0.02 much better than 0.2
            #for i in range(1, 11): #take i*10 precent of train each time

            beta = 1 /avarage_deg
            gamma = 1 / (avarage_deg*avarage_deg)

            _ = main_gcn(adj_matrices=self._adjacency_matrices, X=self._X,
                         labels=self._labels, in_features = self._data.num_features, #num_classes*2- in case we use our 2k vectors,
                         hid_features=25, out_features= num_classes,ds_name = dataSetName, cut= 10*0.1,
                         epochs=100, dropout=0.22705016880182263, lr=0.16979153000000003, l2_pen=0.00090448807155062379,
                         temporal_pen=0.09797882823017063, beta = beta, gamma = gamma,
                         trials=3, dumping_name='',
                         optimizer=Adam,
                         is_nni=self._nni)


        else:
            beta = 1 / avarage_deg
            gamma = 1 / (avarage_deg * avarage_deg)
            _ = main_gcn(adj_matrices=self._adjacency_matrices,
                         labels=self._labels, in_features = num_classes*2,
                         hid_features=int(input_params["hid_features"]), out_features= num_classes,ds_name = dataSetName, cut= 10*0.1,
                         epochs=input_params["epochs"], dropout=input_params["dropout"],
                         lr=input_params["lr"], l2_pen=input_params["regularization"],
                         temporal_pen=0.09797882823017063, beta=beta, gamma=gamma,
                         trials=7, dumping_name='',
                         optimizer=input_params["optimizer"],
                         is_nni=self._nni)
        return None


if __name__ == "__main__":
    # Available features: Degree ('Degree'), In-Degree ('In-Degree'), Out-Degree ('Out-Degree'),
    #                     Betweenness Centrality ('Betweenness'), BFS moments ('BFS'), motifs ('Motif_3', 'Motif_4') and
    #                     the extra features based on the motifs ('additional_features')

    # gcn_detector = GCNCliqueDetector(200, 0.5, 10, True, features=['Motif_3', 'additional_features'],
    #                                  norm_adj=True)
    # gcn_detector.train()
    # gcn_detector = GCNCliqueDetector(500, 0.5, 15, False,
    #                                  features=['Degree', 'Betweenness', 'BFS'], new_runs=0, norm_adj=True)
    # gcn_detector.train()
    # plt.rcParams.update({'figure.max_open_warning': 0})
    gg = GCN_subgraphs()
    gg.train()
    t = 0
    
    
    
    
    
    
    
    
    
    
    
    '''            _ = main_gcn(adj_matrices=self._adjacency_matrices, #X=self._X,
                         labels=self._labels, in_features = num_classes*2,
                         hid_features=40, out_features= num_classes,ds_name = dataSetName, cut= 10*0.1,
                         epochs=40, dropout=0.022311848472689362, lr=0.01296195, l2_pen=0.026338610292950139,
                         temporal_pen=0.09797882823017063, beta = beta, gamma = gamma,
                         trials=3, dumping_name='',
                         optimizer=Adam,
                         is_nni=self._nni)
                         '''
