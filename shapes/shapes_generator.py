import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math




def shapes_generator(nb_triangles, simple_edges):
    ''' generates shapes from specified number of triangles and simples edges
    '''
    M=nb_triangles+simple_edges
    #### generate random connection matrix between triangles, and triangles and simple edges
    G=nx.Graph()
    a=0
    for i in range(nb_triangles):
        G.add_nodes_from([a,a+1,a+2])
        G.add_edges_from([(a,a+1),(a+1,a+2),(a+2,a)])
        a+=1
    for i in range(simple_edges):
        G.add_nodes_from([a,a+1])
        G.add_edges_from([(a,a+1)])
        a+=1
        
    while nx.number_connected_components(G)>1:
    ### randomly merge nodes:
        m=np.random.sample(G.nodes(),2,replace=False)
        G=nx.contracted_nodes(G, m[0], m[1])
    return G
        
    