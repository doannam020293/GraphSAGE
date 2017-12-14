import networkx as nx
from nam_basic.nam_basic import create_connect_to_mongo
import sys
import json
from networkx.readwrite import json_graph
import pandas as pd
import numpy as np
from graphsage.utils import run_random_walks
import os
# db = create_connect_to_mongo()
# coll = db['FbProfiles']


def try_input_file():
    # load thử G.json

    file1 = json.load(open(
        r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\GraphSAGE\example_data\ppi-G.json',
        'r'))
    nodes = file1['nodes']
    nodes = file1['links']
    node = nodes[0]
    type(node['feature'])

    list_node = []
    for node in nodes[0:1000]:
        dict_node = {}
        dict_node['test'] = node.get('test')
        dict_node['val'] = node.get('val')
        list_node.append(dict_node)

    df = pd.DataFrame(list_node)
    df.apply(sum)

    # kiểm tra links
    list_node = []
    for node in nodes[0:1000]:
        dict_node = {}
        dict_node['train_removed'] = node.get('train_removed')
        dict_node['test_removed'] = node.get('test_removed')
        list_node.append(dict_node)

    df = pd.DataFrame(list_node)
    df.apply(sum)

def create_graph():
    '''
    ý tưởng chạy multithread để tạo các graph, sau đó merge các graph lại
    F = nx.compose(G,H)
    :return: 
    '''
    # list_edge = []
    list_friend =coll.find().limit(1000)
    # khoi tạo undirect Graph,
    G = nx.Graph()
    for user in list_friend:
        user_id = user['key']
        for friend in user['friends']:
            neigh_id = 0
            weight = 1
            # list_edge.append([user_id,neigh_id,{'weight':1}])
            G.add_edge([user_id,neigh_id,{'weight':1}])


def create_graph2():
    '''
    ý tưởng: chạy multi thread để tạo các list edge, với weight. sau đó  
    :return: 
    '''
    list_edge = []
    list_friend =coll.find().limit(1000)
    # khoi tạo undirect Graph,
    G = nx.Graph()
    for user in list_friend:
        user_id = user['key']
        for friend in user['friends']:
            neigh_id = 0
            weight = 1
            list_edge.append([user_id,neigh_id,weight])
    G.add_weighted_edges_from(list_edge)

def create_graph3():
    '''
    ý tưởng: query trực tiếp từ data để ra được list (user_id, neigh_id, weight)
    :return: 
    '''
    pipeline =     [{'$project':{
            'key':'$key',
            'friends':'$friends.id', #            'weight':1,
        }}]
    list_f = [[1,2],[1,3]]
    a = (n for n in [1,2,3,5])
    list_friend = list(coll.aggregate(pipeline)) # ta phải convert từ cusor thành list.
    G = nx.Graph()
    G.add_edges_from(list_friend)
    G.add_edges_from(list_f)

def get_multual_friends(G,u,v):
    '''
    lấy số lượng mutual friend between two node u, v in G
    :param G: 
    :param u: 
    :param v: 
    :return: 
    '''
    # G =nx.complete_graph(100)
    # # lấy set các list neighborhood of G
    # a = set(G[1])
    # b = set(G[2])
    # mutual_friend = len(a.intersection(b))
    mutual_friend = len(list(nx.common_neighbors(G, u, v)))
    return mutual_friend

def create_graph_facebook(main_folder = r'C:\nam\work\facebook'):
    '''
    create graph facebook from edge file
    :return: 
    '''
    # G = nx.Graph()
    file = r'C:\nam\work\facebook\facebook_combined.txt'
    G = nx.read_edgelist(file, nodetype=int, create_using=nx.Graph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
        G[edge[0]][edge[1]]['train_removed'] = False
        G[edge[0]][edge[1]]['test_removed'] = False
    for node in G.nodes():
        G.node[node]['val'] = False
        G.node[node]['test'] = False
        # # thêm feature hay không
        # G.node[node]['feature'] = (node,)
    # load graph ra file Graph.json
    with open(os.path.join(file,'face-G.json'), 'w') as outfile1:
        outfile1.write(json.dumps(json_graph.node_link_data(G)))

    #create id_map.json
    nodes = list(G.nodes())
    id_map = {}
    for node in nodes:
        string_id = str(node)
        id_map[string_id] = node
    with open(os.path.join(file,'face-id_map.json'), 'w') as outfile1:
        outfile1.write(json.dumps(id_map))


    # create walk file, chủ yếu là load fucntion  run_random_walks trong utils.py
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    # đoạn random walk này hoàn toàn có thể dung random_walk của node2vec.
    pairs = run_random_walks(G, nodes)
    with open(os.path.join(file,'face-walks1.json'), "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))

    # create class_map file

    nodes = list(G.nodes())
    class_map = {}
    for node in nodes:
        string_id = str(node)
        class_map[string_id] = [1,]
    with open(os.path.join(file,'face-class_map.json'), 'w') as outfile1:
        outfile1.write(json.dumps(id_map))

    #
    # # read numpy embed
    # a = np.load(r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\GraphSAGE\graphsage\unsup-facebook\graphsage_mean_small_0.000010\val.npy')
    # df = pd.DataFrame(a)
    # df.head()

def create_graph_facebook(main_folder = r'C:\nam\work\facebook'):
    '''
    create graph facebook from edge file
    :return: 
    '''
    main_folder = r'C:\nam\work\facebook\facebook_add'
    file = r'C:\nam\work\facebook\facebook network data\facebook_combined.txt'
    G = nx.read_edgelist(file, nodetype=int, create_using=nx.Graph())
    G.add_edge(4040,4039)
    G.add_edge(4041,4042)
    G.add_edge(4043,4044)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
        G[edge[0]][edge[1]]['train_removed'] = False
        G[edge[0]][edge[1]]['test_removed'] = False
    for node in G.nodes():
        G.node[node]['val'] = False
        G.node[node]['test'] = False
        # # thêm feature hay không
        # G.node[node]['feature'] = (node,)
    # load graph ra file Graph.json
    with open(os.path.join(main_folder,'face-G.json'), 'w') as outfile1:
        outfile1.write(json.dumps(json_graph.node_link_data(G)))

    #create id_map.json
    nodes = list(G.nodes())
    id_map = {}
    for node in nodes:
        string_id = str(node)
        id_map[string_id] = node
    with open(os.path.join(main_folder,'face-id_map.json'), 'w') as outfile1:
        outfile1.write(json.dumps(id_map))


    # create walk file, chủ yếu là load fucntion  run_random_walks trong utils.py
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    # đoạn random walk này hoàn toàn có thể dung random_walk của node2vec.
    pairs = run_random_walks(G, nodes)
    with open(os.path.join(main_folder,'face-walks.txt'), "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))

    # create class_map file


