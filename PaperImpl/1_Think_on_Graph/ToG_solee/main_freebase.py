import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queryset', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=256)

    parser.add_argument('--width', type=int, default=3)
    parser.add_argument('--depth', type=int, default=3)

    parser.add_argument('remove_unnecessary_rels', type=bool, default=True)
    
    parser.add_argumnet('--LLM_type', type=str, default='gpt-3.5-turbo')
    parser.add_argumnet('--openai_api_keys', type=str, default='openai_api_keys.json')
    parser.add_argumnet('--num_retain_entity', type=int, default=5)

    parser.add_argument('--prune_tools', type=str, default='llm')


