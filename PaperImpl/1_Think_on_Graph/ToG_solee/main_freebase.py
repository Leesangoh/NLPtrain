import argparse
import os
from tqdm import tqdm
from utils import *
import random

def think_on_graph_using_beamsearch(
                                    ## Queryset name for saving the results
                                    queryset_name, # name of the dataset

                                    ## ToG related arguments
                                    question, # input question
                                    topic_entity, # topic entity
                                    max_length, # max length of LLM's output
                                    prune_tools, # prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert
                                    remove_unnecessary_rels, # whether removing unnecessary relations, XXX: but why do we need this?
                                    
                                    ## Beam search related arguments
                                    width, # beamsearch width
                                    depth, # beamsearch depth

                                    ## LLM related arguments
                                    LLM_type, # used LLM type
                                    openai_api_keys, # openai api keys if using GPT-3.5-turbo or GPT-4 
                                    num_retain_entity, # number of entities retained during entities search
                                    temperature_exploration, # temperature of LLM in exploration stage
                                    temperature_reasoning, # temperature of LLM in reasoning stage   
                                    ):
    
    if len(topic_entity) == 0:
        results = generate_without_explored_paths(question, temperature_reasoning, max_length, openai_api_keys, LLM_type)
        save_2_jsonl(question, results, [], file_name=queryset_name)

    pre_relations = [] # pre-relations are used for not using (1-hop) previous relation
    pre_heads = [-1] * len(topic_entity) # pre-heads are used to indicate whether the entity is head or not, and it is used with pre-relations for search_relations
    all_paths = []
    flag_printed = False

    for cur_depth in range(1, depth + 1):

        print("cur_depth: ", cur_depth)
        print("topic_entity: ", topic_entity)
        
        cur_entity_relations_list = []

        # Search for relations for each entity in the topic entity. That is, Search(x, E^{D-1}, P) and Prune(\pi, x, E^{D-1}_{cand}, P_{cand})
        # E^{D-1} corresponds to topic_entity

        for index, entity in enumerate(topic_entity):
            if entity != "[FINISH_ID]":

                # Search(x, E^{D-1}, P)
                candidate_relations, candidate_head_relations = search_relations(entity, topic_entity[entity], pre_relations, pre_heads[index], question, remove_unnecessary_rels)

                # Prune(\pi, x, E^{D-1}_{cand}, P_{cand})
                retrieved_relations_with_scores = prune_relations(question, entity, topic_entity[entity], candidate_relations, candidate_head_relations, prune_tools, width,
                                                                    temperature_exploration, max_length, openai_api_keys, LLM_type)
                
                cur_entity_relations_list.extend(retrieved_relations_with_scores)

        all_candidates = dict()

        print("After Phase 1, Extracted relations: ", cur_entity_relations_list)

        for index, entity_relation in enumerate(cur_entity_relations_list):

            # Search(x, E^{D-1}, R^D, P)
            entity_candidates_id = search_entities(entity_relation['entity'], entity_relation['relation'], entity_relation['head'])

            ## To reduce the cost of pruning in the next step, we can prune the number of entities to be considered.
            if prune_tools == "llm":
                if len(entity_candidates_id) >= 20:
                    entity_candidates_id = random.sample(entity_candidates_id, num_retain_entity)
            
            ## If there are no entity candidates, then skip the next step
            if len(entity_candidates_id) == 0:
                print("SKIP!!")
                continue
            
            # To further prune the entity candidates, we can use LLM to score each entity. Note that zero-score entities are pruned in this step.
            entity_scores, entity_candidates, entity_candidates_id = score_each_entity(question, entity_candidates_id, entity_relation['score'], entity_relation['relation'], prune_tools, temperature_exploration, max_length, openai_api_keys, LLM_type)

            # Add the entity candidates to the all_candidates
            add_candidates(all_candidates, entity_relation, entity_candidates,  entity_scores, entity_candidates_id)

        # Prune(\pi, x, E^{D-1}_{cand}, P_{cand})
        # Unlike prune_relations, we prune in batch for all entities since we need to maintain the number of entities for each entity
        # by the characteristics of the beam search.

        print("all_candidates: ", all_candidates)

        if 'entity_candidates' not in all_candidates or len(all_candidates['entity_candidates']) == 0:
            reasoning(question, all_paths, cur_depth, temperature_reasoning, max_length, openai_api_keys, LLM_type, no_more_knowledge=True, queryset_name=queryset_name)
            flag_printed = True
            break
    
        flag, paths, entity_ids, pre_relations, pre_heads = prune_entities(all_candidates, width)
        all_paths.append(paths)


        # Reasoning phase
        if flag:
            stop, results = reasoning(question, all_paths, cur_depth, temperature_reasoning, max_length, openai_api_keys, LLM_type, evaluate=True)
            
            if stop:
                print("ToG stopped at depth %d." % cur_depth)
                save_2_jsonl(question, results, all_paths, fname=queryset_name)
                flag_printed = True
                break
            else:
                print("depth %d still not find the answer." % cur_depth)
                flag_finish, entity_ids = if_finish_list(entity_ids)
                if flag_finish:
                    reasoning(question, all_paths, cur_depth, temperature_reasoning, max_length, openai_api_keys, LLM_type, no_more_knowledge=True, queryset_name=queryset_name)
                    flag_printed = True
                else:
                    topic_entity = {entity_id: id_to_entity_name_or_type(entity_id) for entity_id in entity_ids}
                    continue
        else:
            reasoning(question, all_paths, cur_depth, temperature_reasoning, max_length, openai_api_keys, LLM_type, no_more_knowledge=True, queryset_name=queryset_name)
            flag_printed = True

    if not flag_printed:
        print("Generate without explored paths")
        results = generate_without_explored_paths(question, temperature_reasoning, max_length, openai_api_keys, LLM_type)
        save_2_jsonl(question, results, [], fname=queryset_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Beam search related arguments
    parser.add_argument('--width', type=int, default=3)
    parser.add_argument('--depth', type=int, default=3)
    
    # LLM related arguments
    parser.add_argument('--LLM_type', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--openai_api_keys', type=str, default='openai_api_keys.json')
    parser.add_argument('--num_retain_entity', type=int, default=5)
    parser.add_argument('--temparature_exploration', type=float, default=0.4)
    parser.add_argument('--temparature_reasoning', type=float, default=0)
    
    # ToG related arguments
    parser.add_argument('--queryset', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--prune_tools', type=str, default='llm')
    parser.add_argument('--remove_unnecessary_rels', type=bool, default=True)

    args = parser.parse_args()

    question_json_list, question_key = get_questions(args.queryset)

    for question_json in tqdm(question_json_list):
        question = question_json[question_key]
        topic_entity = question_json['topic_entity']

        think_on_graph_using_beamsearch(
            queryset_name=args.queryset,
            question=question,
            topic_entity=topic_entity,
            max_length=args.max_length,
            prune_tools=args.prune_tools,
            remove_unnecessary_rels=args.remove_unnecessary_rels,
            width=args.width,
            depth=args.depth,
            LLM_type=args.LLM_type,
            openai_api_keys=args.openai_api_keys,
            num_retain_entity=args.num_retain_entity,
            temperature_exploration=args.temparature_exploration,
            temperature_reasoning=args.temparature_reasoning
        )