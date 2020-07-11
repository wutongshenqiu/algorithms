### 定义图表示
from array import array
from queue import Queue
from functools import wraps

ARRAY_TYPE = "l"


class AdjList:
    """
    the adjacency list of graph

    attention: the subscript is start from 1

    In this data structure, we use array to store the info
    this array structure is similar to the void CreateListF
    https://wiki.jikexueyuan.com/project/easy-learn-algorithm/clever-adjacency-list.html
    """
    def __init__(self, file_path):
        # we use x, y, z indicate an edge with weights z from x to y
        # and u[i], v[i], w[i] stores the ith edge
        self.u = array(ARRAY_TYPE)
        self.v = array(ARRAY_TYPE)
        self.w = array(ARRAY_TYPE)
        # the subscript is start from 1
        self.u.append(-10000)
        self.v.append(-10000)
        self.w.append(-10000)
        self._read_file(file_path)
        # print(self.u)
        # print(self.v)
        # print(self.w)
        # print(self.next)
        # print(self.first)

    def _read_file(self, file_path):
        with open(file_path, "r", encoding="utf8") as f:
            n, m = map(int, f.readline().split())
            self.n = n
            self.m = m
            # we use array first to store the first edge of each vertex
            # initialize first
            self.first = array(ARRAY_TYPE, (-1 for i in range(n+1)))
            self.next = array(ARRAY_TYPE)
            self.next.append(-10000)
            for count, edge in enumerate(f):
                x, y, z = map(int, edge.split())
                self.u.append(x)
                self.v.append(y)
                self.w.append(z)
                self.next.append(self.first[x])
                self.first[x] = count+1

    def print_edges_of_certain_vertex(self, vertex):
        assert 1 <= vertex <= self.n
        k = self.first[vertex]
        while ~k:
            print(f"{self.u[k]}->{self.v[k]}: {self.w[k]}")
            k = self.next[k]

    def print_edges_of_all_vertex(self):
        for i in range(1, self.n+1):
            k = self.first[i]
            while ~k:
                print(f"{self.u[k]}->{self.v[k]}: {self.w[k]}")
                k = self.next[k]

    def get_edges_of_certain_vertex(self, vertex):
        assert 1 <= vertex <= self.n
        k = self.first[vertex]
        linked_vertices = array(ARRAY_TYPE)
        while ~k:
            linked_vertices.append(self.v[k])
            k = self.next[k]
        return linked_vertices


class Graph:

    def __init__(self, file_path):
        self.adj_list = AdjList(file_path)
        # the number of vertex and edge
        self.n = self.adj_list.n
        self.m = self.adj_list.m
        # predfn stores the pre index of each vertex when in dfs
        self.predfn = array(ARRAY_TYPE, (0 for i in range(self.n+1)))
        # postdfn stores the post index of each vertex when in dfs
        self.postdfn = array(ARRAY_TYPE, (0 for i in range(self.n+1)))

    def construct_adj_matrix(self):
        inf = float("inf")
        self.adj_matrix = [[inf for i in range(self.n+1)] for j in range(self.n+1)]
        for i in range(1, self.m+1):
            u = self.adj_list.u[i]
            v = self.adj_list.v[i]
            w = self.adj_list.w[i]
            self.adj_matrix[u][v] = w

    def dfs_by_stack(self):
        visited = array("b", (0 for i in range(self.n+1)))
        stack = array(ARRAY_TYPE)
        visited[1] = 1
        stack.append(1)
        while len(stack):
            vertex = stack.pop()
            linked_vertices = self.adj_list.get_edges_of_certain_vertex(vertex)
            for linked_vertex in linked_vertices:
                if visited[linked_vertex]:
                    continue
                stack.append(linked_vertex)
                visited[linked_vertex] = 1

    def bfs(self):
        visited = array("b", (0 for i in range(self.n+1)))
        q = Queue()
        visited[1] = 1
        q.put(1)
        while not q.empty():
            vertex = q.get()
            linked_vertices = self.adj_list.get_edges_of_certain_vertex(vertex)
            for linked_vertex in linked_vertices:
                if visited[linked_vertex]:
                    continue
                q.put(linked_vertex)
                visited[linked_vertex] = 1

    def dfs_by_recursive(self):
        Graph.predfn = 1
        Graph.postdfn = 1
        Graph.visited = array("d", (0 for i in range(self.n+1)))
        for i in range(1, self.n+1):
            if not Graph.visited[i]:
                self._dfs_by_recursive(i)
        del Graph.visited
        del Graph.predfn
        del Graph.postdfn

    def _dfs_by_recursive(self, vertex):
        Graph.visited[vertex] = 1
        self.predfn[vertex] = Graph.predfn
        Graph.predfn += 1
        linked_vertices = self.adj_list.get_edges_of_certain_vertex(vertex)
        for linked_vertex in linked_vertices:
            if not Graph.visited[linked_vertex]:
                self._dfs_by_recursive(linked_vertex)
        self.postdfn[vertex] = Graph.postdfn
        Graph.postdfn += 1

    # MST(Minimum Spanning Trees)
    # 包含无向图 G 中所有顶点形成的代价最小的树称之为 mst
    # 最小生成树性质：
    # 将图 G 分成两块，连接两块中权值最小的边一定作为一颗最小生成树上的一条边

    # 显然最小生成树具有最优子结构性质：
    # 即去掉树中的一条边形成的两棵子树都是其对应图的最小生成树

    def prim(self):
        """
            对于每一个节点 v 有两个属性，key 和 pi
            key 表示连接 v 和树的所有边中最小边的权重，
            pi 为其父节点，则算法包含以下几个步骤：
            1. 初始化：所有节点的 key 值为无穷，pi 值为空，
            根节点的 key 值为 0，pi 值为自身
            2. 将所有节点和当前节点直接相连节点的 key 值进行更新
            3. 选择一个最小的 key 的节点加入到树中，并作为当前节点
            4. 重复以上步骤，直到所有的节点都接入到树中

            notations:
                parent: an array to store the parent of each vertex.
                e.g. parent[i] stores the parent node of vertex i.
                key: an array to store the min weight among the edges of each vertex.
                e.g. key[i] stores the min weight of the edges of vertex i.
                visited: an hash set to indicate the nodes are selected in the tree
                edges: an hash set to store the selected edges
                cost: the cost of the tree
            attentions:
                our subscript is started from 1
        """

        if "adj_matrix" not in self.__dict__:
            self.construct_adj_matrix()
        inf = float("inf")
        key = [inf for i in range(self.n+1)]
        parent = [-1 for i in range(self.n+1)]
        visited = set()
        edges = set()
        cost = 0
        # root node
        key[1] = 0
        parent[1] = -100
        visited.add(1)
        # 更新 key
        linked_vertices = self.adj_list.get_edges_of_certain_vertex(1)
        for vertex in linked_vertices:
            key[vertex] = self.adj_matrix[1][vertex]
            parent[vertex] = 1
        while len(visited) < self.n:
            # 获取 not visited 中 key 值最小的顶点
            u = -1
            key_min = float("inf")
            for vertex in range(1, self.n+1):
                if vertex not in visited:
                    if key[vertex] < key_min:
                        key_min = key[vertex]
                        u = vertex
            visited.add(u)
            edges.add((u, parent[u]))
            cost += key[u]

            linked_vertices = self.adj_list.get_edges_of_certain_vertex(u)
            for linked_vertex in linked_vertices:
                if linked_vertex not in visited and self.adj_matrix[u][linked_vertex] < key[linked_vertex]:
                    parent[linked_vertex] = u
                    key[linked_vertex] = self.adj_matrix[u][linked_vertex]

        print(f"the selected edges are: ")
        print(edges)
        print(f"the cost of the tree if: {cost}")

    def kruskal(self):
        """
            1. 初始化：将图 G 中的 n 个顶点看成 n 个孤立的联通分支，将所有的边按权值排序
            2. 选取：遍历所有的边，如果边 (u, v) 属于两个不同的联通分支，则连接成一个联通分支，否则跳过
            3. 重复上述步骤直到只剩下一个联通分支

            notations:
                selected_edges: a hash set to store the selected edges
                sorted_edges: sorted edges
                union_find_set: Union-find set
                cost: the cost of the tree
        """
        if "adj_matrix" not in self.__dict__:
            self.construct_adj_matrix()
        selected_edges = set()
        sorted_edges = sorted(
            [(self.adj_list.u[i], self.adj_list.v[i], self.adj_list.w[i]) for i in range(1, self.m+1)],
            key=lambda a: a[2], reverse=True)

        cost = 0
        union_find_set = UnionFindSet(self.n)
        while len(selected_edges) < self.n-1:
            u, v, w = sorted_edges.pop()
            if union_find_set.find(u) != union_find_set.find(v):
                union_find_set.union(u, v)
                selected_edges.add((u, v))
                cost += w
        print(f"the selected edges are: ")
        print(selected_edges)
        print(f"the cost of the tree if: {cost}")

    # 单元最短路径问题：
    # 给定带权有向图 G，且 G 每条边的权值都是非负的实数
    # 且给定 G 中的一个顶点，称为源，计算从源到其他各顶点的最短路径的长度

    # 单源最短路径具有最优子结构性质

    def dijkstra(self, vertex):
        """
            1. 将顶点划分为两个集合 X、Y，初始化 X 为源点，dis[y] 表示 Y 中的节点经由 X 的节点到源点的最短距离
            2. 选取 dis[y] 最小的顶点 k 加入 X，更新与 k 相邻的顶点的 dis[y]
            3. 重复以上步骤直到 X 含有全部的元素

            notations:
                dis: an array to store the min distance of each vertex to source vertex
                e.g. if the source vertex is 1, dis[2] indicates the min dis between 1 and 2 through the vertex in X
                parent: an array to store the parent of the each vertex
                e.g. parent[i] stores the parent vertex of i
                visited: an array to indicate whether the vertex is visited
        """
        assert 1 <= vertex <= self.n
        if "adj_matrix" not in self.__dict__:
            self.construct_adj_matrix()
        inf = float("inf")
        X = []
        dis = [inf for i in range(self.n+1)]
        parent = [-1 for i in range(self.n+1)]
        visited = [0 for i in range(self.n+1)]

        # add source vertex
        X.append(vertex)
        dis[vertex] = 0
        visited[vertex] = 1
        linked_vertices = self.adj_list.get_edges_of_certain_vertex(vertex)
        for linked_vertex in linked_vertices:
            dis[linked_vertex] = self.adj_matrix[vertex][linked_vertex]

        while len(X) < self.n:
            min_dis = inf
            index = -1
            # find min weight in dis
            for i in range(1, self.n+1):
                if not visited[i]:
                    if dis[i] < min_dis:
                        min_dis = dis[i]
                        index = i
            if not ~index:
                break
            parent[index] = X[-1]
            X.append(index)
            visited[index] = 1
            if ~index:
                linked_vertices = self.adj_list.get_edges_of_certain_vertex(index)
                for linked_vertex in linked_vertices:
                    if dis[index] + self.adj_matrix[index][linked_vertex] < dis[linked_vertex]:
                        dis[linked_vertex] = dis[index] + self.adj_matrix[index][linked_vertex]

        for i in range(1, self.n+1):
            print(f"path from {vertex} to {i}: ", end="")
            reverse_path = []
            t = i
            while ~parent[t]:
                reverse_path.append(str(t))
                t = parent[t]
            reverse_path.append(str(vertex))
            print(f"{'->'.join(reversed(reverse_path))}")
            print(f"cost: {dis[i]}")

    def bellman_ford(self, vertex):
        """
            1. 初始化，将 dis[vertex] = 0，其余为 inf
            2. 遍历每一条边，并作松弛，松弛 i 次，找到经过 i 条边到源点的最短路径
            3. 重复上述操作至多 n-1 次，即可以找到源点到其他点的最短路径
            4. 做第 n 次松弛，如果某一个节点 k 的 dis[k] 值改变，则说明存在负环

            notations:
                dis: an array to stores the min distance between vertex to source vertex
                e.g. if the source vertex is 1, dis[2] means the min distance between 2 to 1
                parent: an array to store the parent of the each vertex
                e.g. parent[i] stores the parent vertex of i
        """
        assert 1 <= vertex <= self.n
        if "adj_matrix" not in self.__dict__:
            self.construct_adj_matrix()
        inf = float("inf")

        dis = [inf for i in range(self.n+1)]
        parent = [-1 for i in range(self.n+1)]

        dis[vertex] = 0
        for i in range(self.n-1):
            for j in range(1, self.n+1):
                linked_vertices = self.adj_list.get_edges_of_certain_vertex(j)
                for linked_vertex in linked_vertices:
                    if dis[linked_vertex] > dis[j] + self.adj_matrix[j][linked_vertex]:
                        dis[linked_vertex] = dis[j] + self.adj_matrix[j][linked_vertex]
                        parent[linked_vertex] = j

        for j in range(1, self.n+1):
            linked_vertices = self.adj_list.get_edges_of_certain_vertex(j)
            for linked_vertex in linked_vertices:
                if dis[linked_vertex] > dis[j] + self.adj_matrix[j][linked_vertex]:
                    print("Exists negative ring!")
                    return False

        for i in range(1, self.n+1):
            print(f"path from {vertex} to {i}: ", end="")
            reverse_path = []
            t = i
            while ~parent[t]:
                reverse_path.append(str(t))
                t = parent[t]
            reverse_path.append(str(vertex))
            print(f"{'->'.join(reversed(reverse_path))}")
            print(f"cost: {dis[i]}")
        return True

    def spfa(self, vertex):
        """
            注意到在 bellman-ford 算法的松弛过程中，我们不需要每次都遍历所有的边
            只需要将与新加入的节点相连的边进行松弛即可以，我们可以用队列来实现这个结构
        """
        assert 1 <= vertex <= self.n
        if "adj_matrix" not in self.__dict__:
            self.construct_adj_matrix()
        inf = float("inf")

        dis = [inf for i in range(self.n + 1)]
        parent = [-1 for i in range(self.n + 1)]
        q = Queue()
        # count 表示某一个点的进队次数，若大于 n 说明有回路
        count = [0 for i in range(self.n+1)]
        dis[vertex] = 0
        count[vertex] += 1

        linked_vertices = self.adj_list.get_edges_of_certain_vertex(vertex)
        for linked_vertex in linked_vertices:
            if dis[linked_vertex] > dis[vertex] + self.adj_matrix[vertex][linked_vertex]:
                dis[linked_vertex] = dis[vertex] + self.adj_matrix[vertex][linked_vertex]
                parent[linked_vertex] = vertex
                count[linked_vertex] += 1
                q.put(linked_vertex)

        while not q.empty():
            j = q.get()
            if count[j] > self.n:
                print("Exists negative ring!")
                return False
            linked_vertices = self.adj_list.get_edges_of_certain_vertex(j)
            for linked_vertex in linked_vertices:
                if dis[linked_vertex] > dis[j] + self.adj_matrix[j][linked_vertex]:
                    dis[linked_vertex] = dis[j] + self.adj_matrix[j][linked_vertex]
                    parent[linked_vertex] = j
                    count[linked_vertex] += 1
                    q.put(linked_vertex)

        for i in range(1, self.n+1):
            print(f"path from {vertex} to {i}: ", end="")
            reverse_path = []
            t = i
            while ~parent[t]:
                reverse_path.append(str(t))
                t = parent[t]
            reverse_path.append(str(vertex))
            print(f"{'->'.join(reversed(reverse_path))}")
            print(f"cost: {dis[i]}")
        return True

    def verify_edge(self):
        """
        use dfs algorithm and divide the edges into four kinds
        思路如下：
            遍历所有的边(假设边为 u->v )，分为这样几种情况：
            1. 如果 v 还没有被访问，则该边是一条树边，并且标记 u 是 v 的父节点
            2. 如果 v 已经被访问，则遍历 u 的所有父亲节点
                - 如果有 v，则为回边
                - 如果没有 v，则遍历 v 的所有父亲节点
                    - 如果有 u，则为前向边
                    - 否则为横跨边
        """
        self.tree_edges = []
        self.forward_edges = []
        self.back_edges = []
        self.cross_edges = []
        Graph.visited = [0 for i in range(self.n+1)]
        Graph.parent = [-1 for i in range(self.n+1)]
        for vertex in range(1, self.n+1):
            if not Graph.visited[vertex]:
                self._verify_edge(vertex, -1)
        del Graph.visited
        del Graph.parent
        print(f"树边：{self.tree_edges}")
        print(f"前向边：{self.forward_edges}")
        print(f"回边：{self.back_edges}")
        print(f"横跨边：{self.cross_edges}")

    def _verify_edge(self, vertex, parent):
        Graph.visited[vertex] = 1
        Graph.parent[vertex] = parent
        linked_vertices = self.adj_list.get_edges_of_certain_vertex(vertex)
        for linked_vertex in linked_vertices:
            if not Graph.visited[linked_vertex]:
                self.tree_edges.append((vertex, linked_vertex))
                self._verify_edge(linked_vertex, vertex)
            else:
                u = vertex
                while ~u:
                    u = Graph.parent[u]
                    if u == linked_vertex:
                        self.back_edges.append((vertex, linked_vertex))
                        return
                v = linked_vertex
                while ~v:
                    v = Graph.parent[v]
                    if v == vertex:
                        self.forward_edges.append((vertex, linked_vertex))
                        return
                self.cross_edges.append((vertex, linked_vertex))



class UnionFindSet:

    def __init__(self, n):
        # 初始化
        self.seq = ["#"]
        # 格式为 (parent, rank)
        self.seq.extend([[-1, 1] for i in range(n)])

    # 找到 p 的祖先节点
    def find(self, p):
        while ~self.seq[p][0]:
            p = self.seq[p][0]
        return p

    def union(self, p, q):
        proot = self.find(p)
        qroot = self.find(q)
        # 祖先节点一样
        if proot == qroot:
            return
        prank = self.seq[proot][1]
        qrank = self.seq[qroot][1]
        if prank == qrank:
            self.seq[proot][1] = self.seq[proot][1] + 1
            self.seq[qroot][0] = proot
        elif prank < qrank:
            self.seq[proot][0] = qroot
        elif prank > qrank:
            self.seq[qroot][0] = proot



if __name__ == '__main__':
    graph = Graph("graph.txt")
    graph.dfs_by_stack()
    graph.bfs()
    # graph.prim()
    graph.kruskal()
    graph.dijkstra(1)
    graph.bellman_ford(1)
    graph.spfa(1)
    graph.verify_edge()