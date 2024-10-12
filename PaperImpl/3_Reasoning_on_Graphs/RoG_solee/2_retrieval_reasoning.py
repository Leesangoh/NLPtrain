import argparse
from datasets import load_dataset
import os
from utils import *
from graph_utils import *
from language_models import get_registed_model
from tqdm import tqdm
import networkx as nx
import random

QUESTION = """Question:\n{question}"""
COT = """ Let's think it step by step."""
GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
CHOICES = """\nChoices:\n{choices}"""
RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""

# Implementation for retrieve and reasoning module
# This module aims to retreive and reason on the graph to answer the question
# That is, given the realation paths (plans), it first retrieves the reasoning paths (shortest paths) from the graph
# Then, it reasons on the reasoning paths to answer the question (P_\theta(a | q, z, G))

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

def retrieve_reasoning_paths(data, processed_list):
    
    graph = build_graph(data['graph'])
    id = data['id']
    q_entity = data['q_entity']
    relation_paths = data['predicted_paths']

    if id in processed_list:
        return None

    reasoning_paths = []

    for entity in q_entity:
        for relation_path in relation_paths:
            reasoning_paths += bfs_with_relation_path(graph, entity, relation_path)

    reasoning_paths = [path_to_string(path) for path in reasoning_paths]

    return reasoning_paths


def reason_on_paths(args, data, model, reasoning_paths):

    question = data['question']
    
    with open(args.prompt_path) as fin:
        prompt_template = f"""{fin.read()}"""

    input = QUESTION.format(question=question)

    if len(data['choices']) > 0:
        input += CHOICES.format(choices = "\n".join(data['choices']))

    prompt_for_length_check = prompt_template.format(instruction=RULE_INSTRUCTION, input = GRAPH_CONTEXT.format(context = "") + input)

    all_tokens = prompt_for_length_check + "\n".join(reasoning_paths)

    if model.tokenize(all_tokens) >= model.maximum_token:
        # we need to use subset of reasoning paths

        random.shuffle(reasoning_paths)
        subset_reasoning_paths = []

        for reasoning_path in reasoning_paths:
            if model.tokenize(prompt_for_length_check + "\n".join(subset_reasoning_paths) + reasoning_path) < model.maximum_token:
                subset_reasoning_paths.append(reasoning_path)
            else:
                break

        reasoning_paths = subset_reasoning_paths
    
    input = prompt_template.format(instruction=RULE_INSTRUCTION, input = GRAPH_CONTEXT.format(context = "\n".join(reasoning_paths)) + input)
    
    return input, model.generate_sentence(input)

def retrieve_and_reason(args, LLM):

    # Load the dataset
    dataset_path = os.path.join(args.datasource_path, args.dataset_name)
    dataset = load_dataset(dataset_path, split='test')

    # Load the relation paths, which is retrieved in step 1_planning.py
    relation_paths = load_jsonl(args.rule_path)
    dataset_with_relation_paths = merge_relation_paths_to_dataset(dataset, relation_paths)

    print("Dataset Loading and Merging Relation Paths Finished")

    output_path = os.path.join(args.outsource_path, args.dataset_name, args.model_name, args.split)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = LLM(args)
    model.prepare_for_inference()

    output_file = os.path.join(output_path, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    for data in tqdm(dataset_with_relation_paths):

        retrieved_paths = retrieve_reasoning_paths(data, processed_list)
        model_input, reasoning_result = reason_on_paths(args, data, model, retrieved_paths)

        total_result = {
            "id": data['id'],
            "question": data['question'],
            "question_entity": data['q_entity'],
            "relation_paths": data['predicted_paths'],
            "predicted_answer": reasoning_result,
            "ground_truth_answer": data['answer'],
            "model_input": model_input
        }

        if total_result is not None:
            fout.write(json.dumps(total_result) + "\n")
            fout.flush()
    
    fout.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasource_path", type=str, default='rmanluo')
    parser.add_argument("--dataset_name", type=str, default="RoG-webqsp")
    parser.add_argument("--outsource_path", type=str, default="results/2_retrieval_reasoning")
    parser.add_argument("--prompt_path", type=str, help="prompt_path", default="prompts/llama2_predict.txt",)
    parser.add_argument("--model_name", type=str, default='RoG') # llama-2-7b-chat-hf
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--rule_path", type=str, default="results/1_planning/RoG-webqsp/RoG/test/predictions_3_False.jsonl")
    parser.add_argument("--force", "-f", action="store_true", help="force to overwrite the results")

    args, _ = parser.parse_known_args()

    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)

    args = parser.parse_args()

    retrieve_and_reason(args, LLM)