from typing import Optional
import numpy as np

from . import mpnn
from .spaces import MetricSpace


class Node:
    """
    A node in the RRT tree structure.

    Args:
        coords (np.ndarray): Coordinates used for nearest neighbor search (typically in configuration space or task space).
        parent (Optional[Node]): Reference to the parent node.
        conf (Any): Optional configuration data associated with the node (can differ from coords).
    """

    def __init__(self, coords: np.ndarray, parent: Optional["Node"], conf=None):
        self.coords = coords
        self.conf = conf
        self.children = []  # pointers to children nodes
        self.parent = parent  # pointer to parent node

    def add_child(self, child):
        self.children.append(child)


class Tree:
    """
    Tree data structure used in RRT-style motion planning algorithms.
    Uses the MPNN library for fast nearest neighbor queries.

    Args:
        metric_space (MetricSpace): The metric space in which nearest neighbor search is performed.
    """

    def __init__(self, metric_space: MetricSpace):
        self.metric_space = metric_space
        self.kd_tree = mpnn.KDTree(metric_space.dim, metric_space.topology, metric_space.scale)

        self.nodes = []

    def add_node(self, coords: np.ndarray, parent: Optional[Node] = None, conf=None) -> Node:
        """
        Adds a node to the tree and connects it to the given parent node.
        The first node added is considered the root and should have `parent=None`.

        Args:
            coords (np.ndarray): Coordinates in the metric space used for nearest neighbor search.
            parent (Optional[Node]): Parent node in the tree.
            conf: Other configuration data associated with the node, if different from `coords`.

        Returns:
            Node: The newly added node.
        """
        new_node = Node(coords, parent, conf)
        if parent is not None:
            parent.add_child(new_node)
        self.nodes.append(new_node)
        self.kd_tree.add_point(coords, len(self.nodes) - 1)
        return new_node

    def get_nearest(self, query_coords: np.ndarray) -> Optional[Node]:
        """
        Finds the node in the tree that is nearest to the given query coordinates.

        Args:
            query_coords (np.ndarray): The query point in the metric space.

        Returns:
            Optional[Node]: The nearest node in the tree, or None if the tree is empty.
        """
        if len(self.nodes) == 0:
            return None

        return self.nodes[self.kd_tree.nearest_neighbor(query_coords)[0]]

    def get_root(self) -> Optional[Node]:
        """
        Returns the root node of the tree (the first added node).

        Returns:
            Optional[Node]: The root node (first added), or None if the tree is empty.
        """
        if len(self.nodes) >= 1:
            return self.nodes[0]

        return None
