from queue import PriorityQueue


def a_star_search(graph, start, goal, heuristic):
    priority_queue = PriorityQueue()
    visited = set()
    
    # Each element in the priority queue is a tuple (total_cost, cost_so_far, current, path)
    priority_queue.put((0 + heuristic(start), 0, start, []))
    
    while not priority_queue.empty():
        total_cost, cost_so_far, current, path = priority_queue.get()
        
        if current == goal:
            path.append(current)
            return path, cost_so_far
        
        if current not in visited:
            visited.add(current)
            
            for neighbor, edge_cost in graph[current]:
                if neighbor not in visited:
                    new_cost_so_far = cost_so_far + edge_cost
                    new_total_cost = new_cost_so_far + heuristic(neighbor)
                    new_path = path + [current]
                    priority_queue.put((new_total_cost, new_cost_so_far, neighbor, new_path))


# Heuristic function
def heuristic(node):
    h_values = {'S': 6, 'A': 3, 'B': 4, 'C': 2, 'D': 2, 'G': 0}
    return h_values[node]


# Graph definition
graph = {
    'S': [('A', 3), ('B', 1)],
    'A': [('S', 3), ('C', 1), ('D', 3), ('G', 4)],
    'B': [('S', 1), ('C', 4)],
    'C': [('A', 1), ('B', 4), ('G', 3)],
    'D': [('A', 3), ('G', 2)],
    'G': [('A', 4), ('C', 3), ('D', 2)]
}


start = 'S'
goal = 'G'


result, total_cost = a_star_search(graph, start, goal, heuristic)
print("Đường đi từ {} đến {}: {}".format(start, goal, result))
print("Chi phí đường đi: {}".format(total_cost))

import networkx as nx
import matplotlib.pyplot as plt


# Create a directed graph
G = nx.DiGraph()
G.add_nodes_from(["S", "A", "B", "C", "D", "G"])
G.add_edges_from([("S", "A"), ("S", "B"), ("B", "C"), ("A", "C"), ("A", "D"), ("A", "G"), ("C", "G"), ("D", "G")])


# Add weights to edges
edge_weights = {("S", "A"): 3, ("S", "B"): 1, ("B", "C"): 4, ("A", "C"): 1, ("A", "D"): 3, ("A", "G"): 4, ("C", "G"): 3, ("D", "G"): 2}
nx.set_edge_attributes(G, edge_weights, 'weight')


# Add heuristic values to nodes
h_values = {'S': 6, 'A': 3, 'B': 4, 'C': 2, 'D': 2, 'G': 0}
nx.set_node_attributes(G, h_values, 'heuristic')


# Shortest path from 'S' to 'A' to 'G'
shortest_path = ['S', 'A', 'G']


# Set 'on_shortest_path' attribute for edges on the shortest path
for u, v in zip(shortest_path[:-1], shortest_path[1:]):
    if G.has_edge(u, v):  # Check if the edge exists in the graph
        G[u][v]['on_shortest_path'] = True


# Set up the plot
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 8))


# Draw edges with straight arrows, color edges on the shortest path red, and set edge labels
edge_colors = ['red' if G[u][v].get('on_shortest_path') else 'black' for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=2, arrows=True, connectionstyle='arc3,rad=0', arrowsize=20)


# Draw nodes without labels
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')


# Draw edge labels
edge_labels = {(u, v): f"{w.get('weight', '')}" for u, v, w in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', label_pos=0.5, rotate=False)


# Draw node labels
node_labels = {node: f"{node}\n(h={G.nodes[node]['heuristic']})" for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels)


plt.axis('off')
plt.show()
