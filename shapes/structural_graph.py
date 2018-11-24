import numpy as np 
import pandas as pd
import networkx as nx

def create_structural_graph(D,k, plot=False):
    ''' INPUT:
    #### D: distance between nodes
    #### k: number of neighbors in the graph
        '''
    G_role=nx.Graph()
    G_role.add_nodes_from(range(D.shape[0]))
    for n in range(D.shape[0]):
        neighbors=np.argsort(D[n,:]).tolist()
        neighbors=neighbors[1:(k+1)]
        print D[n, neighbors]
        for nn in neighbors:
            G_role.add_edge(n,nn)   
    A=nx.adjacency_matrix(G_role)
    
    if plot==True:
        nx.draw_networkx(G_role)
    
    return G_role, A.todense()




def test_graph(G_role,A, indices):
    nb=nx.number_connected_components(G_role)
    Comp=  list(nx.connected_component_subgraphs(G_role))
    for i in range(len(Comp)):
        graph=Comp[i]
        ### check the coherence of the subgraph with respect to the indices
        for j in graph.nodes():
            ###
            print 'nothing'
    Comp=nx.strongly_connected_components(G_role)