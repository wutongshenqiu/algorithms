"""
in the chapter, we learn backtracking
"""

# 回溯法的工作原理
# 回溯法按照深度优先策略搜索问题的解空间树，算法搜索至解空间树的任一节点时
# 1. 判断该节点是否包含问题的解：
#   - 如果不包含：跳过该节点
#   - 如果可能包含：进入该子树，继续按照深度优先策略搜索
#   - 如果某节点的所有子节点都不包含问题的解，回溯到父节点，继续搜索


class QueenProblem:
    """
    example 1: 皇后问题
    在 n * n 的国际象棋上摆放 n 个皇后，使其不能互相攻击，即：
    任意两个皇后都不能处于同一行、同一列或同一斜线上，问摆法数是多少？
    """
    @staticmethod
    def solution(n: int) -> int:
        """
        notations:
            1. we use an one dimension array A to store the place of queen.
            for example, A[2] = 3 means the queen is in the row 3, column 4
            2. we use int k to record the number of the placed queens
        procession:
            1. for each row, we use function check to verify if the solution if valid
            2. if valid:
                - if is an answer, record and backtrack
                - if not an answer, use deep-first search to explore next node
            3. if not valid:
                - if not the end, check next solution
                - if end, backtrack
        """

        A = [-1 for i in range(n)]
        k = 0
        num_of_solutions = 0

        while ~k:
            # record the current place of queen
            A[k] = A[k] + 1
            # check if achieve the end
            if A[k] >= n:
                A[k] = -1
                k -= 1
            else:
                if QueenProblem.check(A, k):
                    k += 1
                    if k >= n:
                        num_of_solutions += 1
                        k -= 1

        return num_of_solutions

    @staticmethod
    def check(A: list, k: int) -> bool:
        """
        check the solution valid or not
        """
        for i in range(k):
            if abs(A[i]-A[k]) == abs(i-k) or A[i] == A[k]:
                return False
        return True


class IntegerDivisionProblem:
    """
    example 2: 整数划分问题
    给定 n 个整数的集合 X = {x_1,x_2,...,x_n}，整数 y，试找到 X 的一个子集 Y，使得 Y 中的所有元素之和等于 y
    """
    from typing import List
    int_vector = List[int]

    @staticmethod
    def solution(X: int_vector, y: int) -> int_vector:
        """
        notations:
            1. we use a int vector B to denote whether the solution vector contains the int in X
            for example, B[5] = -1、0、1 means not initialized、not in Y、in Y respectively
        """
        n = len(X)
        B = [-1 for i in range(n)]
        k = 0

        while ~k:
            B[k] = B[k] + 1
            if B[k] >= 2:
                B[k] = -1
                k -= 1
            else:
                current_sum = IntegerDivisionProblem.check(B, X, y, k)
                if current_sum == y:
                    for i in range(k+1, n):
                        B[i] = 0
                    return B
                elif current_sum < y:
                    k += 1
                if k >= n:
                    k -= 1

    @staticmethod
    def check(B, X, y, k):
        current_sum = 0
        for i in range(k):
            if B[i] == 1:
                current_sum += X[i]
        return current_sum


class ZeroOneBackpack:
    """
    example 3: 0-1 背包问题
    给定 n 种物品和一背包。物品 i 的重量是 w_i，其价值为 v_i，背包的容量为 C，
    应该如何选择装入背包的物品，使得装入背包中物品的总价值最大
    """
    from typing import List
    int_vector = List[int]

    @staticmethod
    def solution(C: int, W: int_vector, V: int_vector) -> int_vector:
        """
        notices:
            1. the answer must be optimal so we must traverse all reasonable possibilities
        notations:
            1. we use an one dimension array B to denote whether solution vector contains the object
            for example, B[4] = 1 indicates the answer contains the 5th object while 0 indicates not
            2. we use an parameter max_value to denote the optimal answers
            3. we use an parameter now_weight to denote the sum of the weights of chosen objects
            4. we use an parameter now_value to denote the sum of the values of chosen objects
        """
        assert len(W) == len(V)
        n = len(W)
        max_B = None
        B = [2 for i in range(n)]
        max_value = 0
        now_weight = 0
        now_value = 0
        k = 0

        while ~k:
            B[k] = B[k] - 1
            if B[k] == 0:
                # subtract the value and the weight
                now_value -= V[k]
                now_weight -= W[k]
            if B[k] <= -1:
                B[k] = 2
                k -= 1
            else:
                if B[k] == 1:
                    if now_weight + W[k] <= C:
                        now_weight += W[k]
                        now_value += V[k]
                    else:
                        B[k] = 0
                k = k + 1

                if k >= n:
                    if max_value < now_value:
                        max_value = now_value
                        max_B = B[:]
                    k -= 1

        return max_B

    @staticmethod
    def optimized_solution(C: int, W: int_vector, V: int_vector) -> int_vector:
        """
        Notices that we have a constrained function to prune the left subtree while right subtree not.
        In order to prune the right subtree, we can use a boundary function. for example, in this
        problem, we can use the answer of decimal backpack, which indicates the optimal case(may not achieve),
        and compare it with the max_value we obtains at present. If the optimal case is smaller,
        than there is no need for us to search next, which means we could just backtrack
        """
        NotImplemented


class MaximumCliqueProblem:
    """
    example 4: 最大团问题
    给定无向图 G=(V,E)，其中 V 是顶点集，E是边集，求 G 中所含定点数最多的完全子图
    """
    from typing import List
    int_matrix = List[List[int]]

    @staticmethod
    def solution(adj_matrix: int_matrix):
        """
        notations:
            max_clique: the optimal answer
        procession:
            1. for the latest valid vertex, verify its adjacent vertex
            2. if valid:
                - if traverse all the vertices, update max_clique
                - else search adjacent vertex
            3. if not valid:
                - use boundary function to judge whether search or not
        """
        NotImplemented


"""
分支界限法 (Branch and Bounds)
1. 常常以广度优先或以最小耗费/最大效益优先的方式搜索解空间树
2. 分支界限法中，每一个活结点只有一次机会称为扩展结点。活结点一旦成为
扩展结点，就一次性产生其所有儿子节点。在这些儿子节点中，导致不可行解的
儿子节点被舍弃，其余儿子节点被加入活结点列表
3. 此后，从活结点列表取下一节点成为当前扩展节点，并重复上述节点扩展过程，
直到找到所需要的解或者活结点列表为空
"""


class TravelSalesmanProblem:
    """
    example 3: 旅行商问题
    求 n 个点形成的无向完全图中经过所有顶点且权值最小的环
    为了简化问题，假定所有的权值均为整数
    """
    from typing import List
    int_matrix = List[List[int]]

    @staticmethod
    def solution(adj_matrix: int_matrix):
        """
        notices:
            the target of TSP is to find the min cost, which indicates
            that we should give our lower bound function

        """
        pass

    @classmethod
    def bound(cls, func_name=""):
        pass

    @staticmethod
    def average_function():
        pass

    @staticmethod
    def greedy(adj_matrix: int_matrix) -> int:
        """
        use greedy function to find a upper bound(this maybe not accurate)
        """
        n = len(adj_matrix)
        visited = [False for i in range(n)]
        cost = 0
        for i in range(n):
            pass


class AssignmentProblem:
    """
    example 4: 任务分配问题
    把 n 项任务分配给 n 个人。每个人完成每项任务的成本不同，
    求分配总成本最小的最优分配方案（为了简便，假设所有的成本为整数）
    """
    from typing import List
    int_matrix = List[List[int]]
    int_list = List[int]

    @classmethod
    def solution(cls, cost_matrix: int_matrix):
        """
        notations:
            cost_matrix: the matrix that stores cost.
            e.g. cost_matrix[i-1][j-1] denotes the cost that
            ith person performs jth task
            min_cost: the optimal cost.
        procession:
            1. use boundary function to find a high bound(maybe not optimal)
            2. initialize the min heap and use evaluation function to find an lower bound
            3. assign each task to 1th person and verify whether within boundary
            4. pop from min heap:
                - if to the end: return an answer
                - else push all adjacent valid in min heap
        """
        upper_bound = cls.bound(cost_matrix, func_name="greedy")
        evaluation_list = cls.evaluate(cost_matrix)
        lower_bound = evaluation_list[0]
        heap = MinHeap()
        # initialize
        n = len(cost_matrix)
        first_person = cost_matrix[0]
        for i in range(n):
            cost = first_person[i]
            evaluation = cost + evaluation_list[1]
            if lower_bound <= evaluation <= upper_bound:
                heap.push(
                    HeapNode(cost=cost, evaluation=evaluation,
                             tasks=[i])
                )

        while not heap.empty():
            node = heap.pop()
            if len(node.tasks) == n:
                print(f"min cost: {node.cost}")
                print(f"tasks: {node.tasks}")
                return
            person = cost_matrix[len(node.tasks)]
            for i in range(n):
                if i in node.tasks:
                    continue
                cost = person[i] + node.cost
                evaluation = cost + evaluation_list[len(node.tasks)+1]
                if lower_bound <= evaluation <= upper_bound:
                    new_tasks = node.tasks.copy()
                    new_tasks.append(i)
                    heap.push(
                        HeapNode(cost=cost, evaluation=evaluation,
                                 tasks=new_tasks)
                    )

    @staticmethod
    def evaluate(cost_matrix: int_matrix) -> int_list:
        """
        in this problem, we just need to use evaluation once
        to generate optimal cost(maybe not achieve)
        """
        n = len(cost_matrix)
        evaluation_list = [0 for i in range(n)]
        evaluation_list[-1] = min(cost_matrix[-1])
        for i in range(n-2, -1, -1):
            evaluation_list[i] = min(cost_matrix[i]) + evaluation_list[i+1]

        evaluation_list.append(0)
        return evaluation_list

    @classmethod
    def bound(cls, cost_matrix: int_matrix, func_name="greedy") -> int:
        if hasattr(cls, func_name):
            return getattr(cls, func_name)(cost_matrix)
        else:
            raise AttributeError(f"{func_name} is not implement!")

    @staticmethod
    def greedy(cost_matrix: int_matrix) -> int:
        n = len(cost_matrix)
        cost = 0
        # record whether the task has been assigned
        tasks_assigned = [False for i in range(n)]
        for i in range(n):
            person = cost_matrix[i]
            min_cost = float("inf")
            min_index = -1
            for j in range(n):
                if not tasks_assigned[j] and person[j] < min_cost:
                    min_cost = person[j]
                    min_index = j
            cost += person[min_index]
            tasks_assigned[min_index] = True

        return cost


from dataclasses import dataclass
from typing import List

int_list = List[int]


@dataclass
class HeapNode:
    cost: int
    evaluation: int
    tasks: List[int]


class MinHeap:
    """
    this min heap should have method pop and push
    """
    def __init__(self):
        self.q = [HeapNode(-1, -1, [-1])]

    def push(self, node: HeapNode):
        self.q.append(node)
        self._shift_up(len(self.q)-1)

    def pop(self):
        self.q[1], self.q[-1] = self.q[-1], self.q[1]
        tmp = self.q.pop()
        if len(self.q) >= 3:
            self._shift_down(1)
        return tmp

    def empty(self):
        return len(self.q) == 1

    def _shift_up(self, index):
        n = len(self.q)
        i = index
        j = index // 2
        while j >= 1:
            if self.q[i].evaluation >= self.q[j].evaluation:
                break
            else:
                self.q[i], self.q[j] = self.q[j], self.q[i]
                i = j
                j = j // 2

    def _shift_down(self, index):
        n = len(self.q)
        i = index
        j = 2 * i
        while j < n:
            if j + 1 < n and self.q[j+1].evaluation < self.q[j].evaluation:
                j = j + 1
            if self.q[i].evaluation <= self.q[j].evaluation:
                break
            else:
                self.q[i], self.q[j] = self.q[j], self.q[i]
                i = j
                j = 2 * j








if __name__ == '__main__':
    # print(ZeroOneBackpack.solution(110, [1, 11, 21, 23, 33, 43, 45, 55], [11, 21, 31, 33, 43, 53, 55, 65]))
    AssignmentProblem.solution([[9, 2, 7, 8],
                                [6, 4, 3, 7],
                                [5, 8, 1, 8],
                                [7, 6, 9, 4]])

