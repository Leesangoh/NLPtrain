import argparse
import wandb
import random
import os
import numpy as np
import torch
from utils.data_classes import ExplaGraphsDataset, SceneGraphsDataset, WebQSPDataset
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from utils.lm_generation import GraphLLM
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import math
import pandas as pd
import re
import string
import json

def get_accuracy_gqa(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        if label in pred:
            correct += 1
    return correct / len(df)


def get_accuracy_expla_graphs(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        matches = re.findall(r"support|Support|Counter|counter", pred.strip())
        if len(matches) > 0 and matches[0].lower() == label:
            correct += 1

    return correct / len(df)


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def get_accuracy_webqsp(path):
    df = pd.read_json(path, lines=True)

    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []

    for prediction, answer in zip(df.pred.tolist(), df.label.tolist()):

        prediction = prediction.replace("|", "\n")
        answer = answer.split("|")

        prediction = prediction.split("\n")
        f1_score, precision_score, recall_score = eval_f1(prediction, answer)
        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)
        prediction_str = " ".join(prediction)
        acc = eval_acc(prediction_str, answer)
        hit = eval_hit(prediction_str, answer)
        acc_list.append(acc)
        hit_list.append(hit)

    acc = sum(acc_list) * 100 / len(acc_list)
    hit = sum(hit_list) * 100 / len(hit_list)
    f1 = sum(f1_list) * 100 / len(f1_list)
    pre = sum(precission_list) * 100 / len(precission_list)
    recall = sum(recall_list) * 100 / len(recall_list)

    print(f"Accuracy: {acc:.4f}")
    print(f"Hit: {hit:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return hit

eval_funcs = {
    "expla_graphs": get_accuracy_expla_graphs,
    "scene_graphs": get_accuracy_gqa,
    "scene_graphs_baseline": get_accuracy_gqa,
    "webqsp": get_accuracy_webqsp,
    "webqsp_baseline": get_accuracy_webqsp,
}

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

load_dataset = {
    'expla_graphs': ExplaGraphsDataset,
    'scene_graphs': SceneGraphsDataset,
    'webqsp': WebQSPDataset,
}

load_model = {
    # "llm": LLM,
    # "inference_llm": LLM,
    # "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    "7b": "meta-llama/Llama-2-7b-hf",
    "7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "13b_chat": "meta-llama/Llama-2-13b-chat-hf",
    "gpt2": "gpt2"

}

def _save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": args,
        "epoch": cur_epoch,
    }
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}_checkpoint_{"best" if is_best else cur_epoch}.pth'
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    torch.save(save_obj, path)


def _reload_best_model(model, args):
    """
    Load the best checkpoint for evaluation.
    """
    checkpoint_path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}_checkpoint_best.pth'

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model

def collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    if 'graph' in batch:
        batch['graph'] = Batch.from_data_list(batch['graph'])
    return batch

def adjust_learning_rate(param_group, LR, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 5e-6
    if epoch < args.warmup_epochs:
        lr = LR * epoch / args.warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    param_group["lr"] = lr
    return lr

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="G-Retriever")

    parser.add_argument("--model_name", type=str, default='graph_llm')
    parser.add_argument("--project", type=str, default="project_g_retriever")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str, default='expla_graphs')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=2)

    # Model Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_steps", type=int, default=2)

    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # LLM related
    parser.add_argument("--llm_model_name", type=str, default='7b')
    parser.add_argument("--llm_model_path", type=str, default='')
    parser.add_argument("--llm_frozen", type=str, default='True')
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--max_txt_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)

    # GNN related
    parser.add_argument("--gnn_model_name", type=str, default='gt')
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)

    args = parser.parse_args()
    
    assert args.model_name == "graph_llm", "Only graph_llm (full model) is supported, since other models are just for ablation study."
    
    # Step 1: Setup WandB
    seed = args.seed
    wandb.init(project=f"{args.project}", name=f"{args.model_name}_{seed}", config=args)
    seed_everything(seed)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]

    print("collate_fn(train_dataset): ", collate_fn(train_dataset[:2]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)

    # Step 4: Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )

    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Step 5: Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):

            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({'Lr': lr})
                wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                accum_loss = 0.

            progress_bar.update(1)

        val_loss = 0
        model.eval()

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch)
                val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
            wandb.log({'Val Loss': val_loss})
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 6: Evaluation
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    print(f'path: {path}')

    model = _reload_best_model(model, args)
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    # Step 7. Post-processing & compute metrics
    acc = eval_funcs[args.dataset](path)
    print(f'Test Acc {acc}')
    wandb.log({'Test Acc': acc})