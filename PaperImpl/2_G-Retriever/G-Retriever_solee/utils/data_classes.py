import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets

# add path to the dataset

PATH = 'dataset'

class ExplaGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.path = f'{PATH}/expla_graphs'
        self.text = pd.read_csv(f'{self.path}/train_dev.tsv', sep='\t')
        self.prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.graph = None
        self.graph_type = 'Explanation Graph'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):

        text = self.text.iloc[index]
        graph = torch.load(f'{self.path}/graphs/{index}.pt')
        question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{self.prompt}'
        nodes = pd.read_csv(f'{self.path}/nodes/{index}.csv')
        edges = pd.read_csv(f'{self.path}/edges/{index}.csv')
        desc = nodes.to_csv(index=False)+'\n'+edges.to_csv(index=False)

        return {
            'id': index,
            'label': text['label'],
            'desc': desc,
            'graph': graph,
            'question': question,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{self.path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{self.path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{self.path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


class SceneGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.path = f'{PATH}/scene_graphs'
        self.prompt = None
        self.graph = None
        self.graph_type = 'Scene Graph'
        self.questions = pd.read_csv(f'{self.path}/questions.csv')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        data = self.questions.iloc[index]
        question = f'Question: {data["question"]}\n\nAnswer:'
        graph = torch.load(f'{self.path}/cached_graphs/{index}.pt')
        desc = open(f'{self.path}/cached_desc/{index}.txt', 'r').read()

        return {
            'id': index,
            'image_id': data['image_id'],
            'question': question,
            'label': data['answer'],
            'full_label': data['full_answer'],
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{self.path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{self.path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{self.path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


class WebQSPDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.path = f'{PATH}/webqsp'
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.q_embs = torch.load(f'{self.path}/q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(f'{self.path}/cached_graphs/{index}.pt')
        desc = open(f'{self.path}/cached_desc/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{self.path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{self.path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{self.path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}