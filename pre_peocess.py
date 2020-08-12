import networkx as nx
import pickle
import numpy as np
import os

# dataSetName = "PubMed"
# num_classes = 3
# avarage_deg  = 4.496018664096972

'DataSets: ' \
'dataSetName = "cora"; num_classes = 7; avarage_deg  = 3.8980797636632203' \
'dataSetName = "CiteSeer"; num_classes = 6; avarage_deg  = 2.7363991584009617' \
'dataSetName = "PubMed"; num_classes = 3; avarage_deg  = 4.496018664096972'

def build_2k_vectors(ds_name, num_classes, train_indices):

    with open(os.path.join("dataSets","gnx_"+ds_name+".pkl"), 'rb') as f:
        gnx = pickle.load(f)
    with open(os.path.join("dataSets","labels_"+ds_name+".pkl"), 'rb') as f:
        labels = pickle.load(f)

    print("start bulding X")
    X = np.zeros((len(gnx), 2 * num_classes))
    X2 = np.zeros((len(gnx), 2))
    for i in range(X.shape[0]):
        # if i%100 == 0:
        #     print("iteration number", i)
        f_neighbors = list(gnx.neighbors(i))
        s_neighbors = []
        for f_neighbor in f_neighbors:
            for s_neighbor in gnx.neighbors(f_neighbor):
                if s_neighbor not in f_neighbors and s_neighbor != i and s_neighbor not in s_neighbors:
                    s_neighbors += [s_neighbor]
        #sub = nx.ego_graph(gnx, 0, radius= 2)
        'part of "if n1 in train_indices " is for making cosideration only for nodes from train, as described in the article)'
        X[i][0:num_classes] = [len([n1 for n1 in f_neighbors if n1 in train_indices and labels[n1] == cls]) for cls in range(num_classes)]
        X[i][num_classes:] = [len([n2 for n2 in s_neighbors if n2 in train_indices and labels[n2] == cls]) for cls in range(num_classes)]
    print("finish bulding X")

    'not needed when loading X for each train set (thats because we 2k vectors, and the neighbors of the 2k should be calculated' \
    'only on nodes from the train set) '
    # with open(os.path.join("dataSets","X_manipulations_"+dataSetName+".pkl"), 'wb') as handle:
    #     pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X


def build_x():
    with open(os.path.join("dataSets","gnx_"+dataSetName+".pkl"), 'rb') as f:
        gnx = pickle.load(f)
    with open(os.path.join("dataSets","labels_"+dataSetName+".pkl"), 'rb') as f:
        labels = pickle.load(f)
    # with open("X_manipulations"+dataSetName+".pkl", 'rb') as f:
    #     X = pickle.load(f)

    print("building X manipulations")
    build_2k_vectors(gnx,labels)

    print("building ego graphs")
    #------------create ego graphs
    ego_graphs = []
    for i in range(len(gnx)):
        sub = nx.ego_graph(gnx,i,radius=2)
        ego_graphs.append(sub)
    with open(os.path.join("dataSets","ego_graphs_"+dataSetName+".pkl"), 'wb') as handle:
        pickle.dump(ego_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # #-------------------check the ego graphs
    # with open(os.path.join("dataSets","ego_graphs.pkl"), 'rb') as f:
    #     egos = pickle.load(f)


    ##-------------------some tries
    # sub = nx.adjacency_matrix(nx.ego_graph(gnx, 0, radius= 2))
    # g=nx.Graph()
    # g.add_edges_from([(5,2),(3,5)])
    # g=nx.adj_matrix(g)
    # g2=nx.from_scipy_sparse_matrix(g)
    # b=3


def check_degree():
    with open(os.path.join("dataSets","gnx_"+dataSetName+".pkl"), 'rb') as f:
        gnx = pickle.load(f)
    with open(os.path.join("dataSets","labels_"+dataSetName+".pkl"), 'rb') as f:
        labels = pickle.load(f)
    return sum([tup[1] for tup in gnx.degree])/len(gnx)



if __name__ == '__main__':

    #build_x()
    # avarage_deg = check_degree()
    # print(avarage_deg)

    b=3
