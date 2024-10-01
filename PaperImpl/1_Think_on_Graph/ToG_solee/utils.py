from prompt_list import *
import json
import openai
from SPARQLWrapper import SPARQLWrapper, JSON
import re

DB_PATH = "http://localhost:8890/sparql"

# Pre-defined sparql

## Search for relations
sparql_search_head_relations_template = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_search_tail_relations_template = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""

## Search for entities
sparql_search_head_entities_template = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_search_tail_entities_template = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 

## Search for entity name or type
sparql_id_to_entity_name_or_type_template ="""PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""

def clean_relations(string, entity_id, head_relations):

    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]

    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})

    if not relations:
        return False, "No relations found"
    
    return True, relations



def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/", "") for relation in relations]

def replace_entity_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]

def execute_sparql(sparql):
    virtuosoDB = SPARQLWrapper(DB_PATH)
    virtuosoDB.setQuery(sparql)
    virtuosoDB.setReturnFormat(JSON)
    results = virtuosoDB.query().convert()
    return results["results"]["bindings"]


def id_to_entity_name_or_type(entity_id):
    sparql = sparql_id_to_entity_name_or_type_template % (entity_id, entity_id)
    results = execute_sparql(sparql)
    if len(results) == 0:
        return "UnName_Entity"
    else:
        return results[0]['tailEntity']['value']

def generate_without_explored_paths(question, temperature_reasoning, max_length, openai_api_keys, LLM_type):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = infer_llm(prompt, temperature_reasoning, max_length, openai_api_keys, LLM_type)
    return response

def infer_llm(prompt, temperature, max_length, openai_api_keys, LLM_type = 'gpt-3.5-turbo'):
    
    # In this version, we only implemented for GPT-3.5-turbo and GPT-4
    # To use Llama, you can easily add the code here

    messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."}, 
                {"role": "user", "content": prompt}]
    
    openai.api_key = openai_api_keys

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=LLM_type,
                messages=messages,
                temperature=temperature,
                max_tokens=max_length,
                frequency_penalty=0,
                presence_penalty=0
            )
            result = response["choices"][0]['message']['content']
            break
        except Exception as e:
            print("Error: ", e)
            continue
    
    return result

def save_2_jsonl(question, answer, reasoning_chains, fname):
    data = {"question": question, "results": answer, "reasoning_chains": reasoning_chains}
    with open("ToG_{}.jsonl".format(fname), "a") as f:
        f.write(json.dumps(data) + "\n")

def construct_relation_prune_prompt(question, entity_name, total_relations, width):
    return extract_relation_prompt % (width, width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

def search_relations(entity_id, entity_name, pre_relations, pre_head, question, remove_unnecessary_rel):

    sparql_search_head_relations = sparql_search_head_relations_template % entity_id
    sparql_search_tail_relations = sparql_search_tail_relations_template % entity_id

    head_relations = execute_sparql(sparql_search_head_relations)
    tail_relations = execute_sparql(sparql_search_tail_relations)

    head_relations = replace_relation_prefix(head_relations)
    tail_relations = replace_relation_prefix(tail_relations)

    def abandon_rels(relation):
        if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
            return True
        return False

    # Remove unnecessary relations (this seems like just engineering)
    if remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]

    if pre_head: # If the previous entity is the head of the relation, then we remove the relations that are already used
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations)) # Remove duplicate relations
    tail_relations = list(set(tail_relations)) # Remove duplicate relations

    all_relations = (head_relations + tail_relations)
    all_relations.sort() # Making the relations in order

    return all_relations, head_relations

def prune_relations(question, entity_id, entity_name, candidate_relations, candidate_head_relations, prune_tools, width,
                    temperature_exploration, max_length, openai_api_keys,
                    LLM_type,
                    ):

    # We only implement for LLM pruning
    if prune_tools != "llm":
        assert False, "Not implemented yet"
    
    prompt = construct_relation_prune_prompt(question, entity_name, candidate_relations, width)
    pruned_relations = infer_llm(prompt, temperature_exploration, max_length, openai_api_keys, LLM_type)
    flag, pruned_relations = clean_relations(pruned_relations, entity_id, candidate_head_relations)

    if flag:
        return pruned_relations
    else:
        return [] # Format error or too small max_length

def search_entities(entity, relation, is_head):

    if is_head:
        # we need to find the tail entities since the relation starts from the head entity
        sparql_search_entities = sparql_search_tail_entities_template % (entity, relation)
    else:
        sparql_search_entities = sparql_search_head_entities_template % (entity, relation)
        
    
    entities = execute_sparql(sparql_search_entities)
    entity_ids = replace_entity_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]

    return new_entity

def are_all_unknown_entities(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)

def remove_unknown_entities(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates

def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)

def score_each_entity(question, entity_candidates_id, score, relation, prune_tools, temperature_exploration, max_length, openai_api_keys, LLM_type):

    print("execute entity_score !!!!!")
    entity_candidates = [id_to_entity_name_or_type(entity_id) for entity_id in entity_candidates_id]

    if are_all_unknown_entities(entity_candidates):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    
    entity_candidates = remove_unknown_entities(entity_candidates)

    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)

    if prune_tools != "llm":
        assert False, "Not implemented yet"
    
    prompt = construct_entity_score_prompt(question, relation, entity_candidates)
    result = infer_llm(prompt, temperature_exploration, max_length, openai_api_keys, LLM_type)

    return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id

def add_candidates(all_candidates, entity_relation, entity_candidates, entity_scores, entity_ids):

    if 'entity_candidates' not in all_candidates:
        all_candidates['entity_candidates'] = []

    if 'entity_scores' not in all_candidates:
        all_candidates['entity_scores'] = []
    
    if 'entity_relations' not in all_candidates:
        all_candidates['entity_relations'] = []
    
    if 'entity_ids' not in all_candidates:
        all_candidates['entity_ids'] = []
    
    if 'topic_entities' not in all_candidates:
        all_candidates['topic_entities'] = []
    
    if 'heads' not in all_candidates:
        all_candidates['heads'] = []

    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_ids = ["[FINISH_ID"]
    
    candidates_relation = [entity_relation['relation']] * len(entity_candidates)
    next_topic_entities = [entity_relation['entity']] * len(entity_candidates)
    is_start_head = [entity_relation['head']] * len(entity_candidates)

    all_candidates['entity_candidates'].extend(entity_candidates)
    all_candidates['entity_scores'].extend(entity_scores)
    all_candidates['entity_relations'].extend(candidates_relation)
    all_candidates['entity_ids'].extend(entity_ids)
    all_candidates['topic_entities'].extend(next_topic_entities)
    all_candidates['heads'].extend(is_start_head)

def prune_entities(all_candidates, width):

    zipped_all_candidates = list(zip(all_candidates['entity_ids'], all_candidates['entity_relations'], all_candidates['entity_candidates'], all_candidates['topic_entities'], all_candidates['heads'], all_candidates['entity_scores']))
    zipped_all_candidates = sorted(zipped_all_candidates, key = lambda x : x[5], reverse = True) # sort by score

    sorted_entity_ids = [x[0] for x in zipped_all_candidates]
    sorted_entity_relations = [x[1] for x in zipped_all_candidates]
    sorted_entity_candidates = [x[2] for x in zipped_all_candidates]
    sorted_topic_entities = [x[3] for x in zipped_all_candidates]
    sorted_heads = [x[4] for x in zipped_all_candidates]
    sorted_entity_scores = [x[5] for x in zipped_all_candidates]

    # slice by width, to get the top width entities
    entity_ids = sorted_entity_ids[:width]
    entity_relations = sorted_entity_relations[:width]
    entity_candidates = sorted_entity_candidates[:width]
    topic_entities = sorted_topic_entities[:width]
    heads = sorted_heads[:width]
    entity_scores = sorted_entity_scores[:width]

    # merge, to prune zero scores
    pruned_candidates = list(zip(entity_ids, entity_relations, entity_candidates, topic_entities, heads, entity_scores))
    pruned_candidates = [x for x in pruned_candidates if x[5] != 0]

    # handle the case where all scores are zero
    if len(pruned_candidates) == 0:
        return False, [], [], [], []
    
    entity_ids, entity_relations, entity_candidates, topic_entities, heads, entity_scores = map(list, zip(*pruned_candidates))
    
    topic_entities_names = [id_to_entity_name_or_type(x) for x in topic_entities] # convert id into name of topic_entities to be saved in the paths
    # we convert into names since paths will be used in reasoning using LLM

    paths = [[(topic_entities_names[i], entity_relations[i], entity_candidates[i])] for i in range(len(entity_candidates))]
    
    return True, paths, entity_ids, entity_relations, heads


def reasoning(question, all_paths, depth, temperature_reasoning, max_length, openai_api_keys, LLM_type, no_more_knowledge=False, queryset_name=None, evaluate=False):

    if no_more_knowledge:
        print(f"No new knowledge added during search depth {depth}, Stop searching.")
    
    if evaluate:
        prompt = evaluate_prompt + question + '\n'
    else:
        prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for path in all_paths for chain in path])
    prompt += "\nKnowledge Triples: " + chain_prompt + 'A: '

    response = infer_llm(prompt, temperature_reasoning, max_length, openai_api_keys, LLM_type)

    def extract_answer(text):
        start_index = text.find("{")
        end_index = text.find("}")
        if start_index != -1 and end_index != -1:
            return text[start_index+1:end_index].strip()
        else:
            return ""
        
    def if_true(prompt):
        if prompt.lower().strip().replace(" ","")=="yes":
            return True
        return False

    if no_more_knowledge:
        save_2_jsonl(question, response, all_paths, fname=queryset_name)
        return
        
    answer = extract_answer(response)

    if if_true(answer):
        return True, response
    else:
        return False, response
    
def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst
    
def get_questions(queryset_name):
    if queryset_name == 'cwq':
        with open('./queryset/cwq.json',encoding='utf-8') as f:
            questions =json.load(f)
        question_string = 'question'
    elif queryset_name == 'webqsp':
        with open('./queryset/WebQSP.json',encoding='utf-8') as f:
            questions =json.load(f)
        question_string = 'RawQuestion'
    elif queryset_name == 'grailqa':
        with open('./queryset/grailqa.json',encoding='utf-8') as f:
            questions =json.load(f)
        question_string = 'question'
    elif queryset_name == 'simpleqa':
        with open('./queryset/SimpleQA.json',encoding='utf-8') as f:
            questions =json.load(f)    
        question_string = 'question'
    elif queryset_name == 'qald':
        with open('./queryset/qald_10-en.json',encoding='utf-8') as f:
            questions =json.load(f) 
        question_string = 'question'   
    elif queryset_name == 'webquestions':
        with open('./queryset/WebQuestions.json',encoding='utf-8') as f:
            questions =json.load(f)
        question_string = 'question'
    elif queryset_name == 'webquestions_sample':
        with open('./queryset/WebQuestions_sample.json',encoding='utf-8') as f:
            questions =json.load(f)
        question_string = 'question'
    elif queryset_name == 'trex':
        with open('./queryset/T-REX.json',encoding='utf-8') as f:
            questions =json.load(f)
        question_string = 'input'    
    elif queryset_name == 'zeroshotre':
        with open('./queryset/Zero_Shot_RE.json',encoding='utf-8') as f:
            questions =json.load(f)
        question_string = 'input'    
    elif queryset_name == 'creak':
        with open('./queryset/creak.json',encoding='utf-8') as f:
            questions =json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    
    return questions, question_string
