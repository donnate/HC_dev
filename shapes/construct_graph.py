#### Set of functions to construct shapes in a network(i.e, subgraphs of a
#### particular shape)

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sc
from shapes import *







def build_lego_structure(list_shapes, start=0,betweenness_density=2.5,plot=False,savefig=False,save2text=''):
    ### This function creates a graph from a list of building blocks by addiing edges between blocks
    #### INPUT:
    #### -------------
    #### width_basis: width (in terms of number of nodes) of the basis
    #### basis_type: (torus, string, or cycle)
    #### list_shapes: list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    G=nx.Graph()
    
    nb_shape=0
    colors=[]         ## labels for the different shapes
    seen_shapes=[]
    seen_colors_start=[]  ## pointer for where should the next shape's labels be initialized
    index_roles=[]  ### roles in the network
    col_start=0
    label_shape=[]
    for shape in list_shapes:
        shape_type=shape[0]
        if shape_type not in seen_shapes:
            seen_shapes.append(shape_type)
            seen_colors_start.append(np.max([0]+index_roles)+1)
            col_start=seen_colors_start[-1]
            ind=len(seen_colors_start)-1
        else:
            ind=seen_shapes.index(shape_type)
            col_start=seen_colors_start[ind]
        start=len(index_roles)
        args=[start]
        args+=shape[1:]
        args+=[col_start]
        S,roles=eval(shape_type)(*args)
        ### Attach the shape to the basis
        G.add_nodes_from(S.nodes())
        G.add_edges_from(S.edges())
        nb_shape+=1
        colors+=[nb_shape]*nx.number_of_nodes(S)
        index_roles+=roles
        label_shape+=[col_start]*nx.number_of_nodes(S)
    ### Now we link the different shapes:
    N=G.number_of_nodes()
    A=np.ones((N,N))
    np.fill_diagonal(A,0)
    for j in np.unique(colors):
        ll=np.array([e==j for e in colors])
        A[ll,:][:,ll]=0
    ### Randomly select edges to put between shapes:
        n=pymc.distributions.rtruncated_poisson(betweenness_density,1)[0]
        start_k=np.array(range(N))[np.array(ll)]
        idx, idy=np.nonzero(A[ll,:])
        indices=np.random.choice(range(len(idx)),n)
        G.add_edges_from([(1+start_k[0]+idx[i],1+idy[i]) for i in indices])

    if plot==True:
        nx.draw_networkx(G,node_color=index_roles,cmap="PuRd")
        if savefig==True:
            plt.savefig("plots/structure.png")
    if len(save2text)>0:
        graph_list_rep=[['Id','shape_id','type_shape','role']]+\
        [[i+1,colors[i],label_shape[i],index_roles[i]] for i in range(nx.number_of_nodes(G))]
        np.savetxt(save2text+"graph_nodes.txt",graph_list_rep,fmt='%s,%s,%s,%s')
        elist=[['Source','Target']]+[[e[0],e[1]] for e in G.edges()]
        np.savetxt(save2text+"graph_nodes.txt",graph_list_rep,fmt='%s,%s,%s,%s')
        np.savetxt(save2text+"graph_edges.txt",elist,fmt='%s,%s')
    return G,colors, index_roles, label_shape
    
    
def build_lego_structure_from_structure(list_shapes, start=0,plot=False,savefig=False,graph_type='nx.connected_watts_strogatz_graph', graph_args=[4,0.4],save2text='',add_node=10):
    ### same as before, except that the shapes are put on top of the spanning tree of a graph
    ### This function creates a graph from a list of building blocks by addiing edges between blocks
    #### INPUT:
    #### -------------
    #### width_basis: width (in terms of number of nodes) of the basis
    #### basis_type: (torus, string, or cycle)
    #### list_shapes: list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    G=nx.Graph()
    
    nb_shape=0
    colors=[]         ## labels for the different shapes
    seen_shapes=[]
    seen_colors_start=[]  ## pointer for where should the next shape's labels be initialized
    index_roles=[]  ### roles in the network
    col_start=0
    label_shape=[]
    
    for shape in list_shapes:
        shape_type=shape[0]
        if shape_type not in seen_shapes:
            seen_shapes.append(shape_type)
            seen_colors_start.append(np.max([0]+index_roles)+1)
            col_start=seen_colors_start[-1]
            ind=len(seen_colors_start)-1
        else:
            ind=seen_shapes.index(shape_type)
            col_start=seen_colors_start[ind]
        start=len(index_roles)
        args=[start]
        args+=shape[1:]
        args+=[col_start]
        S,roles=eval(shape_type)(*args)
        ### Attach the shape to the basis
        G.add_nodes_from(S.nodes())
        G.add_edges_from(S.edges())
        
        colors+=[nb_shape]*nx.number_of_nodes(S)
        index_roles+=roles
        label_shape+=[col_start]*nx.number_of_nodes(S)
        nb_shape+=1
    #print seen_shapes
    ### Now we link the different shapes:
    N=G.number_of_nodes()
    N_prime=nb_shape
    #### generate Graph
    graph_args.insert(0,N_prime+add_node)
    G.add_nodes_from(range(N,N+add_node))
    colors+=[nb_shape+rr for rr in range(add_node)]
    #print colors
    r=np.max(index_roles)+1
    l=label_shape[-1]
    index_roles+=[r]*add_node
    label_shape+=[-1]*add_node
    Gg=eval(graph_type)(*graph_args)
    elist=[]
    ### permute the colors:
    initial_col=np.unique(colors)
    perm=np.unique(colors)
    np.random.shuffle(perm)
    color_perm={initial_col[i]:perm[i] for i in range(len(np.unique(colors)))}
    colors2=[color_perm[c] for c in colors]
    #colors=colors2
    for e in Gg.edges():
        if e not in elist:
            ii=np.random.choice(np.where(np.array(colors2)==(e[0]))[0],1)[0]
            jj=np.random.choice(np.where(np.array(colors2)==(e[1]))[0],1)[0]
            G.add_edges_from([(ii,jj)])
            elist+=[e]
            elist+=[(e[1],e[0])]

    if plot==True:
        nx.draw_networkx(G,node_color=index_roles,cmap="PuRd")
        if savefig==True:
            plt.savefig("plots/structure.png")
    if len(save2text)>0:
        graph_list_rep=[['Id','shape_id','type_shape','role']]+\
        [[i,colors[i],label_shape[i],index_roles[i]] for i in range(nx.number_of_nodes(G))]
        np.savetxt(save2text+"graph_nodes.txt",graph_list_rep,fmt='%s,%s,%s,%s')
        elist=[['Source','Target']]+[[e[0],e[1]] for e in G.edges()]
        np.savetxt(save2text+"graph_nodes.txt",graph_list_rep,fmt='%s,%s,%s,%s')
        np.savetxt(save2text+"graph_edges.txt",elist,fmt='%s,%s')
    return G,colors, index_roles, label_shape




def build_fractal_structure(L,graph_type=[],graph_args=[]):
    '''
    builds a hierarchical_structure
    INPUT
    --------------------------------------------------------------------------------------
    L					: nb of layers
    graph_type			: (list of length L) type of graph at every layer (default:graph_type=['nx.gnp_random_graph']*L )
    graph_args			: params for the clusters at each layers
    OUTPUT
    --------------------------------------------------------------------------------------
    '''
    ####Check if all the arguments have been correctly provided or set to default
    if len(graph_type)!=L or len(graph_args)!=L:
        graph_type=['nx.gnp_random_graph']*L
        graph_args=[[10,0.7]]*L
  
        
    #G0=nx.gnp_random_graph(level0[0],level0[1])
    G0=eval(graph_type[0])(*graph_args[0])
    labels=[0]*G0.number_of_nodes()
    for i in range(1,L):
        G0,labels=build_new_level(G0,labels,[graph_type[i]],graph_args[i])
        #graph_args.remove(0)\
    print('number of connected components:%i'%(nx.number_connected_components(G0) ))  
    return G0,labels
        
        
            
def build_new_level(G0,labels=[],graph_type=[],graph_args=[]):
    '''
    builds a hierarchical_structure
    INPUT
    --------------------------------------------------------------------------------------
    L					: nb of layers
    graph_type			: (list of length L) type of graph at every layer (default:ER)
    nb_nodes			: nb of nodes at each level (list of length L)
    graph_args			: params for the clusters at each layers
    OUTPUT
    --------------------------------------------------------------------------------------
    '''
    G=nx.Graph()
    ####Check if all the arguments have been correctly provided or set to default
    if len(graph_type)==0 or len(graph_args)==0:
    		graph_type=['nx.gnp_random_graph']
    		graph_args=[10,0.7]
    if len(labels)==0:
         labels=[0]*G0.number_of_nodes()
    colors=[]         ## labels for the different shapes
    ### Bottom up construction
    #print(graph_type[0])
    Gg=eval(graph_type[0])(*graph_args)  ####graph structure at new level
    n0=G0.number_of_nodes()
    for i in range(Gg.number_of_nodes()):
         colors+=list(np.array(labels)+i*n0)
         mapping={n: n+i*n0 for n in G0.nodes.keys()}
         G1=nx.relabel_nodes(G0,mapping, copy=True)
         G.add_nodes_from(G1.nodes())
         G.add_edges_from(G1.edges())
    #print(nx.number_connected_components(G0))
    #print('adding links')
    elist=[e for e in G.edges]
    print(elist)
    for e in Gg.edges():
        ii=np.random.choice(np.where( (np.array(colors)//n0)==(e[0]))[0],1)[0]
        jj=np.random.choice(np.where((np.array(colors)//n0)==(e[1]))[0],1)[0]
        if (min([ii,jj]),max([ii,jj])) not in elist:
            #print((min([ii,jj]),max([ii,jj])))
            G.add_edges_from([(ii,jj)])
            elist+=[(min([ii,jj]),max([ii,jj]))]
       
    return G, colors