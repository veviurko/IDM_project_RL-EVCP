import numpy as np
import igraph


def _make_graph(env):
    graph = igraph.Graph()
    pv_color = 'orange'
    gen_color = 'red'
    load_color = 'purple'
    ev_color = 'blue'

    for d_ind, device in enumerate(env.devices):
        if device.type == 'feeder':
            c = gen_color
            s = 12
            shape = 'triangle-down'
            label_size = 10
        elif device.type == 'pv':
            c = pv_color
            s = 12
            shape = 'triangle-up'
            label_size = 15
        elif device.type == 'load':
            c = load_color
            s = 8
            shape = 'square'
            label_size = 15
        elif device.type == 'ev_charger':
            c = ev_color
            s = 12
            shape = 'square'
            label_size = 15
        else:
            raise ValueError(device.type)
        graph.add_vertex(str(d_ind), color=c, size=s, label=device.name, shape=shape, label_dist=2,
                         label_size=label_size, )
    for node_from in range(env.n_devices):
        for node_to in range(node_from, env.n_devices):
            if env.conductance_matrix[node_from, node_to] > 0:
                graph.add_edge(str(node_from), str(node_to), )
    return graph
