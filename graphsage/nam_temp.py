import json
import pandas as pd
a= json.load(open(r'C:\nam\work\temp\data\example_data\ppi-G.json','r'))
type(a)
a.items()
a.keys() # dict_keys(['directed', 'graph', 'nodes', 'links', 'multigraph'])
a['directed']
a['graph']
a['multigraph']

links = a['links']
nodes =a['nodes']
type(nodes)
len(nodes)
node = nodes[2]
node.keys() # dict_keys(['test', 'id', 'feature', 'val', 'label'])
len(node['label'])
sum(node['label'])
len(node['feature'])
node['feature']
sum(node['feature'])

list_node = nodes[1:100]
feature = [a['feature'] for a in list_node]
label= [a['label'] for a in list_node]

df = pd.DataFrame(feature)
df.head()
df.describe()
import numpy as np
df.apply(np.sum,axis=1)
df.apply(np.sum,axis=0)


type(links)
link = links[2]
link.keys() # dict_keys(['test_removed', 'train_removed', 'target', 'source'])


len(node['label'])
sum(node['label'])
len(node['feature'])
link['test_removed']
link['train_removed']

sum(node['feature'])


embed = np.load(r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\GraphSAGE\graphsage\unsup-example_data\graphsage_mean_small_0.000010\val.npy')
embed.shape