import networkx as nx
from collections import deque

def bfs_with_relation_path(graph, starting_node, relation_path):
    
    result_reasoning_paths = []
    queue = deque([(starting_node, [])])

    while queue:

        cur_node, cur_path = queue.popleft()

        if len(cur_path) == len(relation_path):
            result_reasoning_paths.append(cur_path)

        if len(cur_path) < len(relation_path):

            if cur_node not in graph:
                continue
        
            # find neighbors which have the same relation with the next relation in the relation path
            for neighbor in graph.neighbors(cur_node):

                rel = graph[cur_node][neighbor]['relation']

                if rel != relation_path[len(cur_path)] or len(cur_path) > len(relation_path):
                    continue

                queue.append((neighbor, cur_path + [(cur_node, rel, neighbor)]))

    
    return result_reasoning_paths

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    '''
    Get shortest paths connecting question and answer entities.
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p)-1):
            u = p[i]
            v = p[i+1]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)

    return result_paths