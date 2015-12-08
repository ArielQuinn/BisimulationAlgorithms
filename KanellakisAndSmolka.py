# This is Ariel Hoffman's implementation of Kanellakis and Smolka's
# coarse partitioning system.

# To determine whether two machines are bisimilar, we can just find
# both their coarsest partition and then relabel the nodes accordingly.
# If they are not bisimilar, this will not work.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
G = nx.Graph()

class KanellakisAndSmolka:
    def __init__(self, actions):
        self.actions = actions
    
    def getNeighborsGivenAction(self, node, action, Q):
        neighbors = Q.neighbors(node)
        actions = [Q.get_edge_data(node, n)["action"] for n in neighbors]
        trueNeighbors = []
        for neighbor, a in zip(neighbors, actions):
            if(a == action):
                trueNeighbors.append(neighbor)
        return trueNeighbors
    
    # Here, Q is the graph,
    # a is the action
    # Notice that this is many more lines than using the simple splitting
    # Paige and Tarjan take advantage of.
    def split(self,blocks,a,Q,blockIdFromNode):
        # This is where we'll store the new set of blocks.
        newBlocks = []
    
        # Now we go ahead and split every block.
        B1 = []
        B2 = []
        for block in blocks:
            initialNode = block[0] # This should be a node id.
            # Find the initaial neighbors
            neighbors = self.getNeighborsGivenAction(initialNode, a, Q)
            neighborBlocks = [blockIdFromNode[n] for n in neighbors]
            initialSetOfDestBlocks = set(neighborBlocks)
    
            # Now we initialize our two new blocks:
            blockB1 = []
            blockB2 = []
            for node in block:
                neighbors = self.getNeighborsGivenAction(node, a, Q)
                neighborBlocks = [blockIdFromNode[n] for n in neighbors]
                setOfDestBlocks = set(neighborBlocks)
                if(setOfDestBlocks==initialSetOfDestBlocks):
                    blockB1.append(node)
                else:
                    blockB2.append(node)
            B1.append(blockB1)
            if blockB2 != []:
                B2.append(blockB2)
            # We now have B1 and B2 for every block.
    
        # Before we exit, we need to reinitialize our look up dictionary for the block numbers:
        return B1+B2
    
    # This will take the graph of interest and output the desired partition.
    # You also need to supply the actions of interest to you.
    def getCoarsestPartition(self, Q, plot=False):
        blockIdFromNode = dict.fromkeys(Q.nodes(),0)# Each node has a blockID.
        prevLen = 0
        pos = nx.spring_layout(Q)
        blocks = [Q.nodes()]
        while prevLen != len(blocks):
            prevLen = len(blocks)
            for action in self.actions:
                if(plot):
                    self.plotGraph(blocks, Q, pos)
                blocks = self.split(blocks, action, Q, blockIdFromNode)
                for i in xrange(len(blocks)):
                    for node in blocks[i]:
                        blockIdFromNode[node] = i 
        return blocks

    # This plots the graph. The arrows are color coded according to action
    # and the nodes are color coded according to partition. The partition value
    # will depend on the global variable. Initially, there will be only one
    # partition and all the nodes will have the same color.
    def plotGraph(self, blocks, Q, pos=None):
        numOfActions = len(self.actions)
        numOfBlocks = len(blocks)

        plt.figure(1)
        if not pos:
            pos = nx.spring_layout(Q)
        nodeColors = cm.rainbow(np.linspace(0,1,numOfBlocks))
        edgeColors = cm.rainbow(np.linspace(0,1,numOfActions))
        for i in xrange(len(blocks)):
            nx.draw_networkx_nodes(Q,pos,nodelist=blocks[i],node_color=nodeColors[i])
        for i in xrange(numOfActions):
            acts = []
            for edge in Q.edges():
                if(Q.get_edge_data(*edge)["action"]==self.actions[i]):
                    acts.append(edge)
            nx.draw_networkx_edges(Q,pos,edgelist=acts,edge_color=[edgeColors[i]]*len(acts))
        plt.show()
        
    
    # This verifies that the two graphs are bisimilar.
    def isBisimilar(self, Q1, Q2):
        Q = nx.union_all([Q1,Q2], rename=('G-', 'H-'))
        blocks = self.getCoarsestPartition(Q)
        # Confirms that there is at least one of each type of node in each partition.
        # If there is, we have bisimilarity. Otherwise, we don't.
        for block in blocks:
            if not any('H' in nodeName for nodeName in block):
                return False
            if not any('G' in nodeName for nodeName in block):
                return False
        return True
    
       
# Every graph must have edges and edge actions.
# In this case, edge actions will be stored as 

# EXAMPLE 1
# Basic tests:
# The following test is example 1, shown in class
Q = nx.DiGraph()
Q.add_edge(1,2,action=1)
Q.add_edge(1,3,action=2)
Q.add_edge(1,4,action=3)
Q.add_edge(3,4,action=1)
Q.add_edge(3,5,action=2)
actions = [1, 2, 3]

k = KanellakisAndSmolka(actions)
k.getCoarsestPartition(Q, plot=True)

# EXAMPLE 2
# the following is the bisimulation example:
S = nx.DiGraph()
S.add_edge(0,1,action=1)
S.add_edge(1,2,action=2)
S.add_edge(0,2,action=1)
S.add_edge(2,2,action=2)
actions = [1,2]

T = nx.DiGraph()
T.add_edge(0,1,action=1)
T.add_edge(1,1,action=2)

k = KanellakisAndSmolka(actions)
# Verifies Bisimilarity.
print(k.isBisimilar(S,T))
# Demos the coarsest partition functionality again.
Q = nx.disjoint_union_all([S,T])
k.getCoarsestPartition(Q, plot=True)
