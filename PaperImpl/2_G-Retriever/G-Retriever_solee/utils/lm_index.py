# Language model utils for step 1 (indexing)

from tqdm import tqdm
import gensim
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np

pretrained_repo = 'sentence-transformers/all-roberta-large-v1'
batch_size = 256  # Adjust the batch size as needed


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()

        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data

class Sentence_Bert(nn.Module):
    
    def __init__(self, pretrained_repo):
        super(Sentence_Bert, self).__init__()
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

def load_sentence_bert():
    model = Sentence_Bert(pretrained_repo)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    return model, tokenizer, device

def sentence_bert_text_to_embedding(model, tokenizer, device, text):
    
    try:
        encoding = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        dataset = Dataset(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask)

        # DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []

        with torch.no_grad():

            for batch in dataloader:
                # move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # forward pass
                embeddings = model(**batch)

                all_embeddings.append(embeddings)
        
        # concatenate and return the embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

    except:
        return torch.zeros((0, 1024))

    return all_embeddings


load_model = {'sentence_bert': load_sentence_bert}
load_text_to_embedding = {'sentence_bert': sentence_bert_text_to_embedding}