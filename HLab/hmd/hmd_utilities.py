from os.path import join as path_join
from pathlib import Path
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info


class Utilities(object):
    @staticmethod
    def get_basedir():
        return Path(__file__).parent.parent


    @staticmethod
    def get_data_path(dataset: str) -> str:
        return path_join(Utilities.get_basedir(), 'data', dataset)


    @staticmethod
    def save_graph(graph_name, graph, labels, graph_info) -> None:
        graph_path = path_join(Utilities.get_data_path(graph_name), graph_name + '_dgl_graph.bin')
        info_path  = path_join(Utilities.get_data_path(graph_name), graph_name + '_info.pkl')
        
        save_graphs(graph_path, graph, {'labels': labels})
        save_info(info_path, graph_info)

    staticmethod
    def load_graph(graph_name) -> None:
        graph_path = path_join(Utilities.get_data_path(graph_name), graph_name + '_dgl_graph.bin')
        info_path  = path_join(Utilities.get_data_path(graph_name), graph_name + '_info.pkl')
        
        graph, label_dict = load_graphs(graph_path)
        labels = label_dict['labels']
        info = load_info(info_path)

        return graph[0], labels, info
