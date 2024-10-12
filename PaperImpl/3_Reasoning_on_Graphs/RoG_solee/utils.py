import json
import string
import os

def path_to_string(path: list) -> str:
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"
            
    return result.strip()

def merge_relation_paths_to_dataset(qa_dataset, relation_paths, n_proc=1, filter_empty=False):
    question_to_relation_path_mapping = dict()
    for data in relation_paths:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_relation_path_mapping[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = question_to_relation_path_mapping[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_relation_path_mapping[qid]["ground_paths"]
        return sample

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)

    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )

    return qa_dataset

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results

def read_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt_template = f"""{f.read()}"""
    return prompt_template

class InstructFormater(object):
    def __init__(self, prompt_path):
        '''
        _summary_

        Args:
            prompt_template (_type_): 
            instruct_template (_type_): _description_
        '''
        self.prompt_template = read_prompt(prompt_path)

    def format(self, instruction, message):
        return self.prompt_template.format(instruction=instruction, input=message)