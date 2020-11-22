from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional, Callable


class AStar(BestFirstSearch):
    """
    This class implements the Weighted-A* search algorithm.
    A* algorithm is in the Best First Search algorithms family.
    """

    solver_name = 'A*'

    def __init__(self, heuristic_function_type: HeuristicFunctionType, heuristic_weight: float = 0.5,
                 max_nr_states_to_expand: Optional[int] = None,
                 open_criterion: Optional[Callable[[SearchNode], bool]] = None):
        """
        :param heuristic_function_type: The A* solver stores the constructor of the heuristic
                                        function, rather than an instance of that heuristic.
                                        In each call to "solve_problem" a heuristic instance
                                        is created.
        :param heuristic_weight: Used to calculate the f-score of a node using
                                 the heuristic value and the node's cost. Default is 0.5.
        """
        # A* is a graph search algorithm. Hence, we use close set.
        super(AStar, self).__init__(
            use_close=True, max_nr_states_to_expand=max_nr_states_to_expand, open_criterion=open_criterion)
        self.heuristic_function_type = heuristic_function_type
        self.heuristic_function = None
        self.heuristic_weight = heuristic_weight

    def _init_solver(self, problem):
        """
        Called by "solve_problem()" in the implementation of `BestFirstSearch`.
        The problem to solve is known now, so we can create the heuristic function to be used.
        """
        super(AStar, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)
        self.solver_name = f'{self.__class__.solver_name} (h={self.heuristic_function.heuristic_name}, w={self.heuristic_weight:.3f})'

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        Called by solve_problem() in the implementation of `BestFirstSearch`
         whenever just after creating a new successor node.
        Should calculate and return the f-score of the given node.
        This score is used as a priority of this node in the open priority queue.

        TODO [Ex.11]: implement this method.
        Remember: In Weighted-A* the f-score is defined by ((1-w) * cost) + (w * h(state)).
        Notice: You may use `search_node.g_cost`, `self.heuristic_weight`, and `self.heuristic_function`.
        """
        w = self.heuristic_weight
        return (1 - w) * search_node.g_cost + (
                w * self.heuristic_function_type.estimate(self.heuristic_function, search_node.state))

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        """
        Called by solve_problem() in the implementation of `BestFirstSearch`
         whenever creating a new successor node.
        This method is responsible for adding this just-created successor
         node into the `self.open` priority queue, and may check the existence
         of another node representing the same state in `self.close`.

        TODO [Ex.11]: implement this method.
        Have a look at the pseudo-code shown in class for A*. Here you should implement the same in python.
        Have a look at the implementation of `BestFirstSearch` to have better understanding.
        Use `self.open` (SearchNodesPriorityQueue) and `self.close` (SearchNodesCollection) data structures.
        These data structures are implemented in `graph_search/best_first_search.py`.
        Note: The successor_node's g-score is stored under `node.g_cost`. Use it to test whether a better
              solution is found, as done in the pseudo-code taught in class (better solution has a strictly
              smaller g-score).
        Remember: In A*, in contrast to uniform-cost, a successor state might have an already closed node,
                  but still could be improved.
        """
        if successor_node.g_cost == float('inf'):
            return
        already_found_node_with_same_state = self.open.get_node_by_state(successor_node.state)
        if already_found_node_with_same_state is not None:  # A node with state s exists in OPEN
            if successor_node.g_cost < already_found_node_with_same_state.g_cost:  # New Parent is better
                already_found_node_with_same_state.cost = successor_node.cost
                already_found_node_with_same_state.parent_search_node = successor_node.parent_search_node
                already_found_node_with_same_state.expanding_priority = successor_node.expanding_priority
            else:  # old path is better, do nothing
                return

        else:  # state not in open, maybe in CLOSED
            already_found_node_with_same_state = self.close.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state is not None:  # A node with state succ exists in CLOSED
                if successor_node.g_cost < already_found_node_with_same_state.g_cost:  # New parent is better
                    already_found_node_with_same_state.cost = successor_node.cost
                    already_found_node_with_same_state.parent_search_node = successor_node.parent_search_node
                    already_found_node_with_same_state.expanding_priority = successor_node.expanding_priority
                    self.open.push_node(already_found_node_with_same_state)  # move old node from CLOSED to OPEN
                    self.close.remove_node(already_found_node_with_same_state)  # remove from CLOSED
                else:  # old path is better, do nothing
                    return

            else:  # this is a new state - add node to OPEN
                self.open.push_node(successor_node)
