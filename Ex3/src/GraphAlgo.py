import sys
from typing import List
import json

from matplotlib import pyplot as plt

from DiGraph import *
from Node import *
from Edge import *
from src.GraphAlgoInterface import GraphAlgoInterface
from queue import PriorityQueue
import matplotlib.pyplot as plt
import matplotlib.widgets as wgt
import numpy as np


class GraphAlgo(GraphAlgoInterface):

    def __init__(self , g = DiGraph()):
      self.g = g

    """This abstract class represents an interface of a graph."""

    def get_graph(self) -> GraphInterface: #v
        return self.g

    def load_from_json(self, file_name: str) -> bool: #v
        with open(file_name , 'r') as f:
            data = json.load(f)
        for p in data['Nodes']:
            if 'pos' in p:
                self.g.add_node(p['id'] , p['pos'])
            else: self.g.add_node(p['id'])
        for p in data['Edges']:
            self.g.add_edge(p['src'] , p['dest'] , p['w'])
        return True

    def save_to_json(self, file_name: str) -> bool:
        """
        Saves the graph in JSON format to a file
        @param file_name: The path to the out file
        @return: True if the save was successful, False o.w.
        """
        raise NotImplementedError

    def shortest_path(self, id1: int, id2: int) -> (float, list): #v Needs to reverse the path!!
        if id1 not in self.g.nodes: return float('inf'),[]
        if id2 not in self.g.nodes: return float('inf'),[]
        D = {v:float('inf') for v in self.g.get_all_v()}
        # print(D)
        visited = []
        parents = {}
        # print(visited)
        D[id1] = 0
        pq =PriorityQueue()
        pq.put((0 , id1))

        while not pq.empty():
            (dist , currV) = pq.get()
            visited.append(currV)

            for neighbor in self.g.nodes:
                if neighbor == currV: continue
                if (currV , neighbor) in self.g.edges:
                    distance = self.g.edges[(currV , neighbor)].getWeight()
                    if neighbor not in visited:
                        prevCost = D[neighbor]
                        newCost = D[currV] + distance
                        if newCost < prevCost:
                            pq.put((newCost , neighbor))
                            D[neighbor] = newCost
                            parents[neighbor] = currV
        child = id2
        src = id1
        path =[]
        path.append(id2)
        for node in parents:
            parent = parents.get(child)
            path.append(parent)
            if parent == src: break
            child = parent
        if id1 not in path:path =[]
        path.reverse()
        return (D[id2] , path)

    def TSP(self, node_lst: List[int]) -> (List[int], float):
        allContaintsList = []
        for node in node_lst:
            if node not in self.g.nodes.keys(): return None

        allListPathes = []
        for nodeSrc in node_lst:
            for nodeDest in node_lst:
                if nodeSrc == nodeDest: continue
                allListPathes.append(self.shortest_path(nodeSrc , nodeDest)[1])

        for list in allListPathes:
            result = all(elem in list for elem in node_lst)
            if result:
                allContaintsList.append(list)

        if len(allContaintsList) ==0:
            for list1 in allListPathes:
                if len(list1) == 0 : continue
                for list2 in allListPathes:
                    if list1 == list2:continue
                    if len(list1) != 0 and len(list2) !=0:
                        if (list1[len(list1)-1] == list2[0] and list1[0] != list2[len(list2)-1]):
                                allContaintsList.append(list1[0:len(list1)-1] + list2)

        newAllContains = []
        for list in allContaintsList:
            result = all(elem in list for elem in node_lst)
            if result:
                newAllContains.append(list)
        result = newAllContains[0]
        for list in newAllContains:
            if (self.checkWeightOfPath(list) <= self.checkWeightOfPath(result)):
                result = list


        return result , self.checkWeightOfPath(result)

        """
        Finds the shortest path that visits all the nodes in the list
        :param node_lst: A list of nodes id's
        :return: A list of the nodes id's in the path, and the overall distance
        """

    def centerPoint(self) -> (int, float): #v
        pq = {}
        tmpList ={}
        for nodeSrc in self.g.nodes.keys():
                tmpList = self.DjikstraHelper(nodeSrc)
                pq[nodeSrc] =max(tmpList.values())
                tmpList ={}
        if float('inf') in pq.values(): return (-1 , float('inf'))
        resultKey = min(pq , key=pq.get)
        resultFloat = min(pq.values())
        result = (resultKey , resultFloat)
        return result #v

    def plot_graph(self) -> None:

        nodes = self.g.nodes
        edges = self.g.edges
        ids, x, y = [], [], []
        # z=[]
        for v in nodes.values():
            ids.append(v.id)
            vPos = v.pos.split(',')
            x.append(float(vPos[0]))
            y.append(-float(vPos[1]))
            # z.append(v[2])

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        ax.scatter(x, y, c='r')

        for i, txt in enumerate(ids):
            ax.annotate(txt, (x[i], y[i]))

        # for src, dest in edges.keys():
        #     _x, _y, _ = nodes[src].pos.split(',')
        #     _x = float(_x)
        #     _y = float(_y)
        #     _ = float(_)
        #     listSrc = [_x,_y,_]
        #     _t , _p , _q = nodes[dest].pos.split(',')
        #     _t = float(_t)
        #     _p = float(_p)
        #     _q = float(_q)
        #     listDest = [_t , _p , _q]
        #     _dx, _dy, _ = np.array(listDest) - np.array(listSrc)
        #     r = 0.2
        #     x, y = _x + r * _dx, _y + r * _dy
        #     dx, dy = (1 - r) * _dx, (1 - r) * _dy
        #
        #     plt.arrow(x, y, dx, dy, width=5e-5, length_includes_head=True)

        def prnt(self):
            plt.plot([0, 1], [0, 1])

        plt.show()

    def DjikstraHelper(self, id1):
        if id1 not in self.g.nodes: return float('inf'),[]
        D = {v:float('inf') for v in self.g.get_all_v()}
        visited = []
        D[id1] = 0
        pq =PriorityQueue()
        pq.put((0 , id1))

        while not pq.empty():
            (dist , currV) = pq.get()
            visited.append(currV)

            for neighbor in self.g.nodes:
                if neighbor == currV: continue
                if (currV , neighbor) in self.g.edges:
                    distance = self.g.edges[(currV , neighbor)].getWeight()
                    if neighbor not in visited:
                        prevCost = D[neighbor]
                        newCost = D[currV] + distance
                        if newCost < prevCost:
                            pq.put((newCost , neighbor))
                            D[neighbor] = newCost
        return D

    def checkWeightOfPath(self , list): #Check the return value !!!
        result = 0
        for i in range(len(list)):
            edge = self.g.getEdges().get((i,i+1)).getWeight()
            result += edge
        result-=1
        return result