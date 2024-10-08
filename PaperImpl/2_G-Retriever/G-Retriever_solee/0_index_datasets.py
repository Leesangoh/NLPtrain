import argparse
import os
import pandas as pd
import json
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import re
from torch_geometric.data.data import Data
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# custom imports
from utils.lm_index import load_model, load_text_to_embedding


model_name = 'sentence_bert'
path = 'dataset'
dataset = None

def textualize_graph(dataset_name, graph):
        
    if dataset_name == 'expla_graphs':

        triplets = re.findall(r'\((.*?)\)', graph)
        nodes = {}
        edges = []

        for triplet in triplets:
            src, edge_attr, dst = triplet.split(';')
            src = src.lower().strip()
            dst = dst.lower().strip()

            if src not in nodes:
                nodes[src] = len(nodes)
            if dst not in nodes:
                nodes[dst] = len(nodes)
            
            edges.append({'src': nodes[src], 'edge_attr': edge_attr.lower().strip(), 'dst': nodes[dst]})
        
        nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
        edges = pd.DataFrame(edges)

    elif dataset_name == 'scene_graphs':
        
        objectid_to_nodeid = {object_id: idx for idx, object_id in enumerate(graph['objects'].keys())}
        nodes = []
        edges = []

        for object_id, object in graph['objects'].items():
            # nodes
            node_attr = f'name: {object["name"]}'
            x, y, w, h = object['x'], object['y'], object['w'], object['h']
            if len(object['attributes']) > 0:
                node_attr = node_attr + '; attribute: ' + (', ').join(object["attributes"])
            node_attr += '; (x,y,w,h): ' + str((x, y, w, h))
            nodes.append({'node_id': objectid_to_nodeid[object_id], 'node_attr': node_attr})

            # edges
            for rel in object['relations']:
                src = objectid_to_nodeid[object_id]
                dst = objectid_to_nodeid[rel['object']]
                edge_attr = rel['name']
                edges.append({'src': src, 'edge_attr': edge_attr, 'dst': dst})

        nodes = pd.DataFrame(nodes, columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

    elif dataset_name == 'webqsp':
        nodes = {}
        edges = []

        for triplet in graph:
            h, r, t = triplet
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})
        nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

    else:
        raise ValueError('Invalid dataset name')
    
    return nodes, edges

def generate_textualized_graphs(dataset_name):

    global path
    global dataset

    def create_dir(path):
        os.makedirs(f'{path}/nodes', exist_ok=True)
        os.makedirs(f'{path}/edges', exist_ok=True)
        os.makedirs(f'{path}/graphs', exist_ok=True)
        os.makedirs(f'{path}/split', exist_ok=True)

    if dataset_name == 'expla_graphs':

        path = f'{path}/expla_graphs'
        create_dir(path)
        dataset = pd.read_csv(f'{path}/train_dev.tsv', sep='\t')

        for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
            nodes, edges = textualize_graph(dataset_name, row['graph'])
            nodes.to_csv(f'{path}/nodes/{i}.csv', index=False, columns=['node_id', 'node_attr'])
            edges.to_csv(f'{path}/edges/{i}.csv', index=False, columns=['src', 'edge_attr', 'dst'])

    elif dataset_name == 'scene_graphs':

        path = f'{path}/scene_graphs'
        create_dir(path)
        dataset = json.load(open('dataset/gqa/train_sceneGraphs.json'))

        for imageid, object in tqdm(dataset.items(), total=len(dataset)):
            nodes, edges = textualize_graph(dataset_name, object)
            nodes.to_csv(f'{path}/nodes/{imageid}.csv', index=False)
            edges.to_csv(f'{path}/edges/{imageid}.csv', index=False)

    elif dataset_name == 'webqsp':
        path = f'{path}/webqsp'
        create_dir(path)
        dataset = load_dataset("rmanluo/RoG-webqsp")
        dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

        for i in tqdm(range(len(dataset))):
            nodes, edges = textualize_graph(dataset_name, dataset[i]['graph'])
            nodes.to_csv(f'{path}/nodes/{i}.csv', index=False)
            edges.to_csv(f'{path}/edges/{i}.csv', index=False)

    else:
        raise ValueError('Invalid dataset name')


def encode_graph(dataset_name):

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text_to_embedding[model_name]
    
    global path
    global dataset

    if dataset_name == 'expla_graphs':

        # Why there are no questions in this dataset?

        for i in tqdm(range(len(dataset))):
            nodes = pd.read_csv(f'{path}/nodes/{i}.csv')
            edges = pd.read_csv(f'{path}/edges/{i}.csv')

            node_attr = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.LongTensor([edges.src, edges.dst])

            data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
            torch.save(data, f'{path}/graphs/{i}.pt')

    elif dataset_name == 'scene_graphs':

        df = pd.read_csv(f'{path}/questions.csv')

        # encode questions
        q_embs = text2embedding(model, tokenizer, device, df.question.tolist())
        torch.save(q_embs, f'{path}/q_embs.pt')

        # encode graphs
        image_ids = df.image_id.unique()
        for i in tqdm(image_ids):

            nodes = pd.read_csv(f'{path}/nodes/{i}.csv')
            edges = pd.read_csv(f'{path}/edges/{i}.csv')

            node_attr = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.tensor([edges.src, edges.dst]).long()

            pyg_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
            torch.save(pyg_graph, f'{path}/graphs/{i}.pt')
        

    elif dataset_name == 'webqsp':
        
        dataset = load_dataset("rmanluo/RoG-webqsp")
        dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        questions = [i['question'] for i in dataset]

        # encode questions
        q_embs = text2embedding(model, tokenizer, device, questions)
        torch.save(q_embs, f'{path}/q_embs.pt')

        # encode graphs
        for i in tqdm(range(len(dataset))):

            nodes = pd.read_csv(f'{path}/nodes/{i}.csv')
            edges = pd.read_csv(f'{path}/edges/{i}.csv')
            nodes.node_attr.fillna("", inplace=True)

            node_attr = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

            pyg_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
            torch.save(pyg_graph, f'{path}/graphs/{i}.pt')

    else:
        raise ValueError('Invalid dataset name')

def generate_split(dataset_name):
    
    # Split the data into train, val, and test sets

    global path
    global dataset

    if dataset_name == 'expla_graphs':
        
        indices = np.arange(len(dataset))
        train_indices, temp_data = train_test_split(indices, test_size=0.4, random_state=42)
        val_indices, test_indices = train_test_split(temp_data, test_size=0.5, random_state=42)

        # Save the indices to separate files
        with open(f'{path}/split/train_indices.txt', 'w') as file:
            file.write('\n'.join(map(str, train_indices)))

        with open(f'{path}/split/val_indices.txt', 'w') as file:
            file.write('\n'.join(map(str, val_indices)))

        with open(f'{path}/split/test_indices.txt', 'w') as file:
            file.write('\n'.join(map(str, test_indices)))

    elif dataset_name == 'scene_graphs':
        
        questions = pd.read_csv(f"{path}/questions.csv")

        unique_image_ids = questions['image_id'].unique()

        # Shuffle the image IDs
        np.random.seed(42)
        shuffled_image_ids = np.random.permutation(unique_image_ids)

         # Split the image IDs into train, validation, and test sets
        train_ids, temp_ids = train_test_split(shuffled_image_ids, test_size=0.4, random_state=42)  # 60% train, 40% temporary
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)  # Split the 40% into two 20% splits

        # Create a mapping from image ID to set label
        id_to_set = {image_id: 'train' for image_id in train_ids}
        id_to_set.update({image_id: 'val' for image_id in val_ids})
        id_to_set.update({image_id: 'test' for image_id in test_ids})

        # Map the sets back to the original DataFrame
        questions['set'] = questions['image_id'].map(id_to_set)

        # Create the final train, validation, and test DataFrames
        train_df = questions[questions['set'] == 'train']
        val_df = questions[questions['set'] == 'val']
        test_df = questions[questions['set'] == 'test']

        # Writing the indices to text files
        train_df.index.to_series().to_csv(f'{path}/split/train_indices.txt', index=False, header=False)
        val_df.index.to_series().to_csv(f'{path}/split/val_indices.txt', index=False, header=False)
        test_df.index.to_series().to_csv(f'{path}/split/test_indices.txt', index=False, header=False)

    elif dataset_name == 'webqsp':

        dataset = load_dataset("rmanluo/RoG-webqsp")
        
        train_indices = np.arange(len(dataset['train']))
        val_indices = np.arange(len(dataset['validation'])) + len(dataset['train'])
        test_indices = np.arange(len(dataset['test'])) + len(dataset['train']) + len(dataset['validation'])

        # Save the indices to separate files
        with open(f'{path}/split/train_indices.txt', 'w') as file:
            file.write('\n'.join(map(str, train_indices)))

        with open(f'{path}/split/val_indices.txt', 'w') as file:
            file.write('\n'.join(map(str, val_indices)))

        with open(f'{path}/split/test_indices.txt', 'w') as file:
            file.write('\n'.join(map(str, test_indices)))

    else:
        raise ValueError('Invalid dataset name')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='expla_graphs', help='dataset name: expla_graphs, scene_graphs, webqsp')

    args = parser.parse_args()

    if args.dataset not in ['expla_graphs', 'scene_graphs', 'webqsp']:
        raise ValueError('Invalid dataset name')

    # Note that these three functions must be call in order
    generate_textualized_graphs(args.dataset)
    encode_graph(args.dataset)
    generate_split(args.dataset)