### 贪心算法

# 贪心算法总是做出在当前看来最好的选择
# 即，贪心算法并不从整体最优考虑，它所作出的选择只是在某种意义下的局部最优
# 贪心算法不一定对所有问题都得到整体最优解，但对许多问题能产生局部最优解
# 如单源最短路径问题，最小生成树问题
# 在一些情况下，贪心算法不能得到整体最优解，其最终结果是最优解的近似



### 活动安排问题：
# 给出 n 个活动 A = {a_1,a_2,a,...,a_n}
# 每个活动的起始时间和结束时间分别是 (s_i,f_i)
# 选出最大的相容活动子集

# https://blog.csdn.net/ii1245712564/article/details/45420061#4.3

class ActivitySelection:

    def greedy(self, S, F):
        '''
            greedy method to solve activity selection problem

        :param S: start time list
        :param F: finish time, assume F is a ascend list
        :return:
        '''

        # the number of activity
        n = len(S)
        # store selected activity
        selected_activity = []

        # every time we choose the earliest end activity whose start time later than the end time of pre activity
        selected_activity.append(0)

        finish_time = F[0]
        for i in range(1, n):
            if S[i] >= finish_time:
                selected_activity.append(i)
                finish_time = F[i]
            else:
                continue
        return selected_activity


# the code has problem
########################################################################################
    # def dp(self, S, F):
    #     '''
    #     introduction:
    #         use dynamic programming to solve activity problem
    #
    #     thought:
    #         for each activity a_k, the question is divided to S_ij_1, S_j2_j, a_k
    #         j_1 < k and compatible with a_k, j_2 > k and compatible with a_k
    #
    #     Args:
    #         S: start time list
    #         F: finish time, assume F is a ascend list
    #
    #     return:
    #
    #     '''
    #
    #     assert len(S) == len(F)
    #
    #     # the number of activities
    #     n = len(S)
    #
    #     # c is the optimal result between a_i and a_j
    #     c = [[0 for i in range(n)] for j in range(n)]
    #     # initialize c
    #     for i in range(n):
    #         c[i][i] = 1
    #
    #     for t in range(2, n-1):
    #         for i in range(n):
    #             for k in range(i+1, i+t):
    #                 left = right = k
    #                 # start time and finish time of activity k
    #                 s_k = S[k]
    #                 f_k = F[k]
    #                 # find left boundry
    #                 while left > i and F[left] > s_k:
    #                     left -= 1
    #                 while right < i+t and S[right] < f_k:
    #                     right += 1
    #
    #
    #
    #     return c[0][n-1]
#################################################################################################3

    @staticmethod
    def test():
        test_case = ActivitySelection()
        test1_S = [1, 3, 0, 5, 3, 5, 6, 8, 8, 2, 12]
        test1_F = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        print(test_case.greedy(test1_S, test1_F))


### 霍夫曼编码
# 霍夫曼编码通常用于数据压缩
# 霍夫曼编码满足以下条件
# 1. 前缀码：没有任何码字是其他码字的前缀
# 2. 编码位将每个码字连接
# 3. 由于没有重复的前缀，因此解码的时候不会发生冲突


class HuffmanNode:
    """
        the node of huffman tree
    """

    def __init__(self, freq, depth=0, value=None, left=None, right=None):
        self.freq = freq
        # depth 指的是以该节点位根节点的子树的高度，并非是该节点深度
        self.depth = depth
        self.left = left
        self.right = right
        self.value = value

    def merge(self, other_node):
        assert isinstance(other_node, HuffmanNode)
        new_freq = self.freq + other_node.freq
        new_depth = max(self.depth, other_node.depth) + 1

        # not sure about the influence other order of the child tree
        new_node = HuffmanNode(
            freq=new_freq, depth=new_depth, left=self, right=other_node
        )
        return new_node


class HuffmanHeap:
    """
        huffman small heap
    """
    def __init__(self, huff_nodes):
        all((isinstance(tree, HuffmanNode) for tree in huff_nodes))
        self.seq = ["#"]
        self.seq.extend(huff_nodes)

    # 堆中元素的上移操作
    def shift_up(self, i):
        # 如果是根节点
        if i < 1 or i >= len(self.seq):
            raise Exception("Invalid index i!")
        if i == 1:
            return
        while i > 1:
            if self.seq[i].freq >= self.seq[i//2].freq:
                break
            else:
                self.seq[i], self.seq[i//2] = self.seq[i//2], self.seq[i]
            i = i // 2

    def shift_down(self, i):
        # 如果是叶子节点
        if i < 1:
            raise Exception("Invalid index i!")
        if 2 * i >= len(self.seq):
            return
        while 2 * i < len(self.seq):
            j = 2 * i
            if j + 1 < len(self.seq) and self.seq[j+1].freq < self.seq[j].freq:
                j = j + 1
            if self.seq[i].freq <= self.seq[j].freq:
                break
            self.seq[i], self.seq[j] = self.seq[j], self.seq[i]
            i = j

    def insert(self, x):
        self.seq.append(x)
        self.shift_up(len(self.seq)-1)

    def delete(self, index):
        last = self.seq.pop()
        # 如果 pop 的是最后一个元素
        if index == len(self.seq):
            return
        tmp = self.seq[index]
        self.seq[index] = last
        if last.freq <= tmp.freq:
            self.shift_up(index)
        else:
            self.shift_down(index)

    def pop(self):
        last = self.seq[1]
        self.delete(1)
        return last

    def make_heap(self):
        for i in range((len(self.seq)-1)//2, 0, -1):
            self.shift_down(i)

    def construct_huffman_tree(self):
        while True:
            if len(self.seq) == 2:
                return self.seq[1]
            x = self.pop()
            y = self.pop()
            self.insert(x.merge(y))

    # DLR
    def show_code(self, root, code):
        if not root.left and not root.right:
            print(f"{root.value}: {''.join(code)}")
        if root.left:
            self.show_code(root.left, code+"0")
        if root.right:
            self.show_code(root.right, code+"1")

    @staticmethod
    def test():
        test1 = [("f", 5), ("e", 9), ("c", 12), ("b", 13), ("d", 16), ("a", 45)]
        test1_huff_trees = [
            HuffmanNode(
                value=value,
                freq=freq
            ) for value, freq in test1
        ]
        huffman_heap = HuffmanHeap(test1_huff_trees)
        huffman_heap.make_heap()
        tree = huffman_heap.construct_huffman_tree()
        huffman_heap.show_code(tree, "")


### 用动态规划解决 huffman 编码问题
# 似乎类似于 OBST(optimal binary search tree)
# 区别在于，huffman 编码只能够在叶子节点搜索到值

### 详细思路
# 1. 显然 huffman tree 具有最优子结构，即 huffman tree 的子树必然也是 huffman tree
# 2. 注意到 huffman 编码的 cost 即为所有非叶子节点的权重之和，huffman 即为所有非叶子节点的权重之和最小
# 3. 若用 e[i][j] 表示包含包含节点 i-j 的哈夫曼树的***非叶子节点***的权重之和
#    e[i][j] = min(e[i][i]+e[i+1][j]+w[i+1][j], e[i][j-1]+e[j][j]+w[i][j-1], e[i][k]+e[k][j]+w[i][k]+w[k][j])
#    w[i][j] = w[i][k]+w[k][j]
# 4. 用 w[i][j] 表示包含节点 i-j 的哈夫曼树的当前节点的值

class HuffmanTree:

    def construct_huffman_by_dp(self, F, V):
        """
        introduction:
            use dynamic programming to construct huffman tree
        Notations:

        Args:
            :param F: the frequency of each value
            :param V: the value
        :return:
        """

        # the number of node
        n = len(F)
        # initialize w
        w = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            w[i][i] = F[i]
        # initialize e
        e = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            e[i][i] = F[i]

        for k in range(1, n):
            for i in range(0, n-k):
                if k == 1:
                    e[i][i+k] = e[i][i]+e[i+k][i+k]
                    w[i][i+k] = e[i][i+k]
                    continue
                cost = float("inf")
                index = -1
                for j in range(i, i+k+1):
                    if j == i:
                        now_cost = e[i][j] + e[j+1][i+k] + w[j+1][i+k]
                    elif j == i+k:
                        now_cost = e[i][j-1] + e[j][j] + w[i][j-1]
                    else:
                        now_cost = e[i][j] + e[j][i+k] + w[i][j] + w[i][i+k]
                    if now_cost < cost:
                        cost = now_cost
                        index = j
                e[i][i+k] = cost
                w[i][i+k] = w[i][index-1] + w[index][i+k]
                # root[i][i+k] = index

        return e[0][n-1]

    @staticmethod
    def test():
        huffman = HuffmanTree()
        test1_F = [5, 9, 12 ,13, 16, 45]
        test1_V = ['f', 'e', 'c', 'b', 'd', 'a']
        huffman.construct_huffman_by_dp(test1_F, test1_V)


#### 最小生成树
# 若 G = (V, E) 是无向联通带权图，即一个网络
# E 中每条边的权为 c[v][w]
# 若 G 的子图 G' 是一颗包含 G 的所有顶点的树，则称 G' 是 G 的生成树
# 生成树上各边权的总和成为该生成树的耗费，耗费最小的生成树称为最小生成树
### 一个实际的例子为，在某地分布着 N 个村庄，需要在 N 个村庄之间修路，
# 问如何修最短的路将各个村庄之间连接起来

### 我的想法：大致可以理解为找一条最短的路径，该路径将图中所有的点都联通

#### 介绍的两种算法：Prim 算法和 Kruskal 算法都利用下如下最小生成树的性质：
# 设G=(V,E)是连通带权图，U是V的真⼦子集。如果
# (u,v)∈E，且u∈U，v∈V-U，且在所有这样的边中，
# (u,v)的权c[u][v]最小，那么一定存在G的一棵最小⽣生成
# 树，它以(u,v)为其中⼀一条边

## 显然最小生成树满足最优子结构性质：即移去 MST T 中的一条边形成的两个子树 T1，T2 也是 MST

#### Prim 算法
# 1. 将图分为两类 V，U
# 2. 初始化一个节点加入 V
# 3. 每次向 V 中加入距离最短的节点
# 4. 更新距离

class MST:

    def prim(self, G):
        """
            use prim algorithm to find MST of G
        :param G: the 2d list store the distance of each vertex
        :return:
        """
        NotImplemented




if __name__ == "__main__":
    # ActivitySelection.test()
    # HuffmanHeap.test()
    HuffmanTree.test()