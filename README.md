# Directed weighted graph implemantation python.

In this assigment we implement classes such as Graph and graph algorithms such as:  
* Center
* TSP
* Dijkstra
* And a GUI interface of graph drawing using MatPlotLib


## Explanation about the Class:
 
 ## The Directed Weighted Graph class
The main purpose of this class is to demonstarte how a graph is really look like.
We desided not to implement a class of nodes and edges like in the last assigment, in the Graph constructor we created two
dicts(but in aim to initialize the graph we give it two lists in the json files) , one for nodes and one for edges, and with them we implemented all the other functions.

The main veriables which we use in this class are:
  - nodes (dict type)
    - here we held all our nodes using the uniqe ids and position
  - edges (dict type)
    - here we held all our edges using three variables:
      - src : what is the node id i came from
      - dest : what is the node id i go to
      - w: what is the "weight" of this edge

|Main Functions|Explenation|RunTime|
|---|---|---|
|DiGraph constructor | The main constuctor of new Graph, here we initialize the nodes dict and the edges dict |O(1)
|get_all_v| return as a dict all the node in the graph |O(1)
|all_out_edges_of_node| return a dict of all the edges which go out from a given node |O(1)
|all_in_edges_of_node| return a dict of all the edges which in to a given node |O(1)
|add_node| adding a new node to our data structure(if his id exsist)|O(1)
|add_edge| create new edge in the graph|O(1)
|remove_node| in this function we removing a node from our graph, we need to find the givven id delete the edge go from it and into it|O(E)
|remove_edge| here we remove an edge from our graph|O(1)
|v_size| gives us the amount of nodes in our graph|O(1)
|e_size| gives us the number of edges in the graph|O(1)


## The GraphAlgo class
The main purpose of this class is to implement known graph algorythms on the graph we was created.
We liked this part because we did the same thing in java but in more complicated and long way,
and now we really apply the approch of KIS- Keep It Simple

The main veriables which we use in this class are:
  * graph (which is a DiGraph type)
    * here we have all of the graph information we built earlier.
   
|Main Functions|Explenation|RunTime|
|---|---|---|
|GraphAlgo constructor | The main constuctor of the Graph's algorythms, using the DiGraph class|O(1)|
|get_graph| getting the graph information|O(1)|
|isConnected| this function check if a graph is strongly connected using the DFS method and the Kosaraju algorythm, this function is Boolean and we implement it in aim to solve the center algo |O(V +V*(V+E))|
|shortestpath| in this function i have used in the Dijkstra algorythm, at first we update al the nodes "weight" to inf execept the src node which we update to 0, we ran on all the nodes and update their weight to the min, and also saved the prev node of each node in the shortest path in aim to return the list of nodes we walk in |O(V^2 +E)|
|center| in the center function we find the center of a graph (if it connected) using the shortestPathDist, for each node we check the max path to other node and take the maximum one, than we returned the shortest path between all of the nodes maximum path and take his node | O(V^4)
|TSP| in a given list of cities we need to find the shortest path which contain all of them and the weight of it|O(n^4)
|load_from_json| this function loads the json file to our project|O(V+E)|
|
