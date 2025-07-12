from typing import Dict, Iterable, List, Optional, Union, Any

from clownpiece.tensor import Tensor, ones_like, zeros_like
from clownpiece.utils_ import wrap_tuple
from collections import deque
from queue import Queue
import threading

"""
    Autograd Module
"""

# autograd/autograd.py
class Node():
    node_id: int
    next_edges: List["Edge"]

    def __init__(self):
        self.node_id = None
        self.next_edges = []
        
    def run(self, *args, **kargs):
        raise NotImplementedError("run method not implemented for abstract Node instance")
    
    # define __hash__ and __eq__ to use Node as dict's key
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id

class Edge():

    input_nr: int # the Edge points to the i-th input of target Node
    node: Optional[Node] # target node the Edge points to

    def __init__(self, input_nr: int, node: Optional[Node]):
        self.input_nr = input_nr
        self.node = node
    
    @staticmethod
    def gradient_edge(tensor: Tensor) -> "Edge":
        # your implement here
        
        # case 0: not a tensor (for convenience)
        if not isinstance(tensor, Tensor):
            return Edge(0, None)

        # case 1: tensor is not a leaf tensor -> use it's grad_fn and output_nr
        if tensor.grad_fn is not None:
            return Edge(tensor.output_nr, tensor.grad_fn)

        # case 2: tensor is a leaf tensor and requires grad -> AccumulateGrad Function
        elif tensor.requires_grad:
            from .function import AccumulateGrad
            return Edge(0, AccumulateGrad(tensor))

        # case 3: tensor is a leaf tensor and requires no grad -> node = None
        else:
            return Edge(0, None)
        

class GraphRoot(Node):
    """
    Root node in the computation graph.
    """

    def __init__(self, tensor: Tensor, grad: Tensor):
        # your implement here
        
        # step1. store the grad
        super().__init__()
        self.tensor = tensor
        self.grad = grad
        
        # step2. create a single edge points to tensor.grad_fn
        self.next_edges = [Edge.gradient_edge(tensor)]
    
    def run(self, *args, **kargs):
        # your implement here

        # step1. return the stored grad
        return self.grad

class NodeTask():
    """
    NodeTask wraps a Node and all its input. 
    It's a ready-to-run Node in GraphTask.
    """

    base: "GraphTask"
    node: Node
    inputs: List[Tensor]
    
    def __init__(self, node: Node, inputs: List[Tensor], base: "GraphTask"):
        self.base = base
        self.node = node
        self.inputs = inputs
        
    def run(self):
        # your implement here
        
        # step1. run the node with inputs
        grads = self.node.run(*self.inputs)
        grads = wrap_tuple(grads)

        # step2. fill the input buffer in GraphTask
        for edge, grad in zip(self.node.next_edges, grads):
            if edge is not None and edge.node is not None:
                self.base.fill_input(edge.node, grad, edge.input_nr)


class GraphTask():
    
    """
    GraphTask wraps the execution of a computation graph.
    """
    
    roots: List[Node] # GraphRoots instances
    nodes: List[Node] # all nodes in the computation graph
    dependencies: Dict[Node, int] # count of inbound degree for topological sort
    inputs_buffer: Dict[Node, List[Tensor]] # inputs_buffer to accumulate intermediate results.
    
    def __init__(self, roots: List[Node]):
        roots = wrap_tuple(roots)
        roots = [root for root in roots if root is not None]
        
        if not roots:
            raise ValueError("roots is empty")
    
        self.roots = roots
        self.nodes = []
        self.dependencies = {}
        self.inputs_buffer = {}
        self._construct_graph()
        self.lock = threading.Lock()
        
    # helper function to assign node_id and initialize self.nodes, dependencies and inputs_buffer
    def _construct_graph(self):
        # your implement here
        id_cnt = 0
        queue = deque(self.roots)
        
        while queue:
            node = queue.popleft()
            if node is None or node.node_id is not None:
                continue
            
            self.nodes.append(node)
            id_cnt += 1
            node.node_id = id_cnt
            
            for edge in node.next_edges:
                if edge is not None and edge.node is not None:
                    queue.append(edge.node)
        
        self.dependencies = {node : 0 for node in self.nodes}
        max_inputs = {node : -1 for node in self.nodes}
        
        for node in self.nodes:
            for edge in node.next_edges:
                if edge is not None and edge.node is not None:
                    self.dependencies[edge.node] += 1
                    max_inputs[edge.node] = max(max_inputs[edge.node], edge.input_nr)
        
        self.inputs_buffer = {node : [None] * (max_inputs[node] + 1) for node in self.nodes}
        
        
    # execute
    def run(self):
        # your implement here
        self._run_single_thread()
        # self._run_multi_thread()

    # for debug
    def _run_single_thread(self):
        # your implement here

        # perform topological sort to execute the graph
        queue = deque()
        for node in self.nodes:
            if self.dependencies[node] == 0:
                queue.append(NodeTask(node, (), self))

        # while queue is not empty:
        while queue:
            # 1. node_task = queue.pop()
            node_task = queue.popleft()
            # 2. node_task.run()
            node_task.run()
            
            for edge in node_task.node.next_edges:
                if edge is not None and edge.node is not None:
                    # 3. decrement dependencies count for target nodes of outbound edges
                    self.dependencies[edge.node] -= 1
                    # 4. enqueue a new NodeTask if dependencies drops to zero. (remember to delete the node in inputs_buffer to release memory.)
                    if self.dependencies[edge.node] == 0:
                        inputs = self.inputs_buffer.pop(edge.node)
                        queue.append(NodeTask(edge.node, inputs, self))

    # for production
    def _run_multi_thread(self):
        # your implement here

        # step1. maintain a shared ready queue for NodeTasks=
        ready_queue = Queue()

        # step2. def a worker function, similar to _run_single_thread.
        # be careful: do not use `while queue is not empty` as exit condition directly. (why?)
        
        def worker():
            nonlocal ready_queue, exceptions, completed
            try:
                while completed < len(self.nodes):
                    try:
                        node_task = ready_queue.get(block=False)
                    except:
                        break
                    
                    node_task.run()
                    
                    with self.lock:
                        for edge in node_task.node.next_edges:
                            if edge is not None and edge.node is not None:
                                self.dependencies[edge.node] -= 1
                                if self.dependencies[edge.node] == 0:
                                    inputs = self.inputs_buffer.pop(edge.node)
                                    ready_queue.put(NodeTask(edge.node, inputs, self))
                    completed += 1
                    ready_queue.task_done()
            except Exception as exc:
                exceptions.append(exc)
        
        exceptions = []
        completed = 0
        
        for node in self.nodes:
            if self.dependencies[node] == 0:
                ready_queue.put(NodeTask(node, (), self))
        
        # step3. spawn multiple worker threads.
        num_threads = 4
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # step4. wait for threads to join.
        ready_queue.join()
        for t in threads:
            t.join()
        
        if exceptions:
            raise Exception(exceptions)
                    
    # accumulate input_grad to self.inputs_buffer[node][input_nr]
    def fill_input(self, node: Node, input_grad: Tensor, input_nr: int):
        # your implement here
        with self.lock:
            if self.inputs_buffer[node][input_nr] is None:
                self.inputs_buffer[node][input_nr] = input_grad
            else:
                self.inputs_buffer[node][input_nr] = self.inputs_buffer[node][input_nr] + input_grad


"""
    Execute backward pass.    
"""
def backward(tensors: Union[Tensor, List[Tensor]], grads: Optional[Union[Tensor, List[Tensor]]] = None):
    tensors = wrap_tuple(tensors)

    if grads is None:
        grads = [ones_like(tensor) for tensor in tensors]
    grads = wrap_tuple(grads)
    
    # wrap with GraphRoots
    graph_roots = [
        GraphRoot(tensor, grad) for tensor, grad in zip(tensors, grads) if tensor.requires_grad
    ]
    
    # if not graph_roots:
    #     raise ValueError("No tensors with requires_grad=True found. Cannot perform backward pass.")

    # execute with GraphTask
    gt = GraphTask(graph_roots)
    gt.run()