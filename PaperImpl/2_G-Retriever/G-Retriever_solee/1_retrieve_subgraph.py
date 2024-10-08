import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data.data import Data
import numpy as np
from pcst_fast import pcst_fast

path = 'dataset'

def retrieval_via_pcst(graph, q_embs, nodes, edges, topk_n=3, topk_e=3, cost_e=0.5):

    assert topk_n > 0, "topk_n must be greater than 0"
    assert topk_e > 0, "topk_e must be greater than 0"

    # corner case
    if len(nodes) == 0 or len(edges) == 0:
        desc = nodes.to_csv(index=False) + '\n' + edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        return graph, desc
    
    # Step 1: Retrieve top-k nodes, and set their prizes. If topk element, set prize to cosine similarity, else set to 0

    n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_embs, graph.x)
    topk_n = min(topk_n, graph.num_nodes)
    _, topk_n_indices = torch.topk(n_prizes, topk_n, largest=True)
    n_prizes = torch.zeros_like(n_prizes)
    n_prizes[topk_n_indices] = torch.arange(topk_n, 0, -1).float()

    # Step 2: Retrieve top-k edges, and set their prizes. If topk element, set prize to cosine similarity, else set to 0

    c = 0.01

    e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_embs, graph.edge_attr)
    topk_e = min(topk_e, e_prizes.unique().size(0))
    topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
    e_prizes[e_prizes < topk_e_values[-1]] = 0.0

    # Question: Why do we perform in the following way?
    # It seems like it is trying to make the prizes unique, and then re-distribute the prizes to the elements that have the same value.
    # Is it because there are lots of duplicate values in the prizes?
    # I don't know any other reason why we would perform this operation.

    last_topk_e_value = topk_e

    for k in range(topk_e):
        indices = e_prizes == topk_e_values[k]
        value = min((topk_e-k)/sum(indices), last_topk_e_value)
        e_prizes[indices] = value
        last_topk_e_value = value * (1 - c)
    
    # reduce the cost of the edges such that at least one edge is selected
    cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))

    # Step 3: Construct the PCST problem based on the prizes and costs

    costs_pcst = []
    edges_pcst = []
    virtual_edges_pcst = []
    virtual_costs_pcst = []

    virtual_n_prizes_pcst = [] # virtual nodes

    mapping_n = {}
    mapping_e = {}

    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]

        if prize_e <= cost_e: 
            # positive cost in PCST problem. We just add reduced edge cost (i.e., cost_e - prize_e as a cost of an edge)
            mapping_e[len(edges_pcst)] = i # mapping from edge index to original edge index, to be used for re-constructing the subgraph
            edges_pcst.append((src, dst))
            costs_pcst.append(cost_e - prize_e)
        else: 
            # negative cost in PCST problem. Since PCST does not support negative costs, we add virtual nodes and edges.
            # virtual node has prize_e - cost_e as a prize
            # virtual edge has 0 cost
            # thus, selecting virtual node and virtual edge is equivalent to selecting the original edge

            virtual_node_id = graph.num_nodes + len(virtual_n_prizes_pcst) # add unique id for virtual node
            mapping_n[virtual_node_id] = i # mapping from virtual node id to original edge index, to be used for re-constructing the subgraph
            virtual_edges_pcst.append((src, virtual_node_id))
            virtual_edges_pcst.append((virtual_node_id, dst))
            virtual_costs_pcst.append(0)
            virtual_costs_pcst.append(0)
            virtual_n_prizes_pcst.append(prize_e - cost_e)

    prizes_pcst = np.concatenate([n_prizes, np.array(virtual_n_prizes_pcst)]) # node prizes for all nodes (including virtual nodes)
    num_edges = len(edges_pcst)

    if len(virtual_costs_pcst) > 0: # means that there are virtual nodes and edges
        costs_pcst = np.array(costs_pcst + virtual_costs_pcst)
        edges_pcst = np.array(edges_pcst + virtual_edges_pcst)

    # Step 4: Solve the PCST problem

    root = -1
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0

    nodes_pcst, edges_pcst = pcst_fast(edges_pcst, prizes_pcst, costs_pcst, root, num_clusters, pruning, verbosity_level)

    # Step 5: Reconstruct the selected subgraph

    selected_nodes = nodes_pcst[nodes_pcst < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges_pcst if e < num_edges]
    selected_virtual_nodes = nodes_pcst[nodes_pcst >= graph.num_nodes]

    print("selected_nodes: ", selected_nodes)
    print("selected_edges: ", selected_edges)
    print("selected_virtual_nodes: ", selected_virtual_nodes)

    if len(selected_virtual_nodes) > 0:
        selected_virtual_edges = [mapping_n[i] for i in selected_virtual_nodes] # find original edge
        selected_edges = np.array(selected_edges + selected_virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()])) # add virtual nodes

    n = nodes.iloc[selected_nodes]
    e = edges.iloc[selected_edges]
    desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc

def retrieve_subgraph(dataset_name):

    if dataset_name == 'expla_graphs':
        # We do nothing, because we exploit all graphs in the dataset
        # Due to the small size of the dataset, we can afford to load all graphs at once.
        # That is, we do not need to perform PCST.
        return

    global path
    path = f'{path}/{dataset_name}'
    
    path_nodes = f'{path}/nodes'
    path_edges = f'{path}/edges'
    path_graphs = f'{path}/graphs'

    cached_graph = f'{path}/cached_graphs'
    cached_desc = f'{path}/cached_desc'

    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)

    if dataset_name == 'scene_graphs':

        questions = pd.read_csv(f'{path}/questions.csv')
        q_embs = torch.load(f'{path}/q_embs.pt')

        for index in tqdm(range(len(questions))):

            if os.path.exists(f'{cached_graph}/{index}.pt'):
                continue
            image_id = questions.iloc[index]['image_id']
            graph = torch.load(f'{path_graphs}/{image_id}.pt')
            nodes = pd.read_csv(f'{path_nodes}/{image_id}.csv')
            edges = pd.read_csv(f'{path_edges}/{image_id}.csv')
            subg, desc = retrieval_via_pcst(graph, q_embs[index], nodes, edges, topk_n=3, topk_e=3, cost_e=0.5)
            torch.save(subg, f'{cached_graph}/{index}.pt')
            open(f'{cached_desc}/{index}.txt', 'w').write(desc)

    elif dataset_name == 'webqsp':

        dataset = load_dataset("rmanluo/RoG-webqsp")
        dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        q_embs = torch.load(f'{path}/q_embs.pt')

        print("dataset loading done")
        print("dataset: ", dataset)
        for index in tqdm(range(len(dataset))):\

            if os.path.exists(f'{cached_graph}/{index}.pt'):
                continue
            graph = torch.load(f'{path_graphs}/{index}.pt')
            nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
            edges = pd.read_csv(f'{path_edges}/{index}.csv')
            q_emb = q_embs[index]
            subg, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk_n=3, topk_e=5, cost_e=0.5)
            torch.save(subg, f'{cached_graph}/{index}.pt')
            open(f'{cached_desc}/{index}.txt', 'w').write(desc)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='expla_graphs', help='dataset name: expla_graphs, scene_graphs, webqsp')

    args = parser.parse_args()

    if args.dataset not in ['expla_graphs', 'scene_graphs', 'webqsp']:
        raise ValueError('Invalid dataset name')

    retrieve_subgraph(args.dataset)
    