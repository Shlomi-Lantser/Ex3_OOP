# from abc import ABC , abstractmethod

from GraphInterface import *
from Node import *
from Edge import *


class DiGraph(GraphInterface):
    def __init__(self, nodes={}):
        self.nodes = {}
        self.edges = {}
        self.mc = 0

    def v_size(self) -> int: #v
        return len(self.nodes)

    def e_size(self) -> int: #v
        return len(self.edges)

    def get_all_v(self) -> dict: #v
        return self.nodes

    def all_in_edges_of_node(self, id1: int) -> dict: #v
        if id1 not in self.nodes.keys():
            return
        result = {}
        for e in self.edges.values():
            # print(e.getSrc())
            if (e.getDst() == id1):
                result[e.getSrc()] = e.getWeight()
        return result

    def all_out_edges_of_node(self, id1: int) -> dict: #v
        if id1 not in self.nodes.keys():
            return
        result = {}
        for e in self.edges.values():
            if (e.getSrc() == id1):
                result[e.getDst()] = e.getWeight()
        return result

    def get_mc(self) -> int: #v
        return self.mc

    def add_edge(self, id1: int, id2: int, weight: float) -> bool: #v
        edge = Edge(id1 , id2 , weight)
        for e in self.edges.values():
            if (e.getSrc() == id1 and e.getDst() == id2):
                return False
        self.edges[(id1, id2)] = edge
        return True

    def add_node(self, node_id: int, pos: tuple = None) -> bool: #v
        node = Node(node_id, pos)
        if pos:
            node.pos = tuple((float(n) for n in node.pos.split(',')))
        if node_id in self.nodes.keys(): return False
        self.nodes[node_id] = node
        # self.nodes[node.id].pos = tuple((float(n) for n in node['pos'].split(',')))
        self.mc += 1
        return True

    def remove_node(self, node_id: int) -> bool: #X
        if node_id not in self.nodes.keys():
            return False
        copy = self.edges.copy()
        del self.nodes[node_id]
        for e in copy.values():
            if (e.getSrc() == node_id):
                del self.edges[(node_id , e.getDst())]
            elif (e.getDst() == node_id):
                del self.edges[(e.getSrc() , node_id)]
        self.mc+=1
        return True

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
        if (node_id1 , node_id2) not in self.edges.keys():
            return False
        del self.edges[(node_id1 , node_id2)]
        self.mc += 1
        return True

    def getEdges(self):
        return self.edges
