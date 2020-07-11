from pprint import pprint


# 动态规划原理

# 具备两个要素:
# 1. 最优子结构
# 2. 子问题重叠

# 最优子结构：最优解一定包含子结构的最优解
# 子问题重叠：子问题空间必须足够小

# 0,1 背包问题：
# 给定 n 种物品和一背包，物品 i 的重量是 w_i，其价值为 v_i，背包的容量为 C
# 求应该如何选择装入背包的物品，使得装入背包物品的总价值最大

# 递归关系式
# 定义 m(i, j) 表示可选择前 i 个物品，背包容量为 j 的情况下的最优值
# 则 m(i, j) = m(i-1, j) if j <= w_i or max(m(i-1, j), m(i-1, j-w_i)+v_i) if j >= w_i

# 0-1 背包问题的优化：
# m(i, j) 中不同值的个数远小于 j
# 二维数组 m[i][j] 中有很多重复的值，因此只需要记录跳跃点
# 用 p[i] 存储 m[i][j] 的全部跳跃点(j, m(i, j))
# 用 q[i-1] 记录 (j+w_i, m(i-1, j)+v_i)
# 则由上述递归的定义可知，p[i] 可由 p[i-1] 和 q[i-1] 推出

class ZeroOneKnapsack:

    def solve_by_2d_array(self, C, W, V, n):
        '''
        use 2d array to solve knapsack problem

        :param C: the capacity of knapsack
        :param W: the list of the weights of each items
        :param V: the list of the values of each items
        :param n: the number of the items
        :return: the optimal solution
        '''

        m = [[0 for i in range(C+1)] for j in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, C+1):
                # if j >= w_i
                if j >= W[i-1]:
                    m[i][j] = max(m[i-1][j], m[i-1][j-W[i-1]]+V[i-1])
                else:
                    m[i][j] = m[i-1][j]
        # use for debug
        print(m)
        return m[n][C]



    # !!! 合并 p[i-1] 和 q[i-1] 的步骤有一些问题
    def solve_by_1d_array(self, C, W, V, n):
        p = [[] for i in range(n+5)]
        p[0].append((0, 0))

        for i in range(1, n+1):
            # 将 p[i-1] 和 q[i-1] 合并成 p[i]
            tmp_p = p[i-1]
            q = []
            # make q in accordance with p[i-1]
            for t in range(len(tmp_p)):
                j, m_ij = tmp_p[t]
                q.append((j+W[i-1], m_ij+V[i-1]))
            # make p[i]
            # 另一种思路：
            # 先按照 j 的大小进行合并，再进行处理
            # 1. 用一个列表存储第一次合并之后的数据
            # 2. 依次遍历列表里面的数据

            # 第一次处理的列表
            pre_p = []
            p_pointer = 0
            q_pointer = 0
            now_j = q[q_pointer][0]
            while True:
                while p_pointer < len(tmp_p):
                    p_j, m_p_ij = tmp_p[p_pointer]
                    if p_j <= now_j:
                        pre_p.append((p_j, m_p_ij))
                        p_pointer += 1
                    else:
                        now_j = p_j
                        break
                while q_pointer < len(q):
                    q_j, m_q_ij = q[q_pointer]
                    if q_j <= now_j:
                        pre_p.append((q_j, m_q_ij))
                        q_pointer += 1
                    else:
                        now_j = q_j
                        break
                if p_pointer == q_pointer == len(tmp_p):
                    break
            # 第二次处理 pre_p
            i1 = 0
            i2 = 1
            p[i].append(pre_p[i1])
            while i2 < len(pre_p):
                i1_j, m_i1_ij = pre_p[i1]
                while i2 < len(pre_p):
                    i2_j, m_i2_ij = pre_p[i2]
                    if i2_j > C:
                        i2 += 1
                        continue
                    if m_i2_ij > m_i1_ij:
                        i1 = i2
                        p[i].append(pre_p[i1])
                        break
                    i2 += 1

        return p[n][-1][1]

# codes below has some problems
###################################################################################
        #     p_pointer = 0
        #     p_j_now, m_p_now_ij = tmp_p[p_pointer]
        #     q_pointer = 0
        #     q_j_now, m_q_now_ij = q[q_pointer]
        #     while True:
        #         while p_pointer < len(tmp_p):
        #             p_j, m_p_ij = tmp_p[p_pointer]
        #             if p_j <= q_j_now:
        #                 p[i].append((p_j, m_p_ij))
        #                 if m_p_ij > m_q_now_ij:
        #                     if q_pointer < len(q):
        #                         q_pointer += 1
        #                         q_j_now, m_q_now_ij = q[q_pointer]
        #             else:
        #                 p_j_now, m_p_now_ij = tmp_p[p_pointer]
        #                 break
        #             p_pointer += 1
        #
        #         while q_pointer < len(q):
        #             q_j, m_q_ij = q[q_pointer]
        #             if (q_j <= p_j_now or p_pointer == len(tmp_p)) and q_j <= C:
        #                 p[i].append((q_j, m_q_ij))
        #                 if m_q_ij > m_p_now_ij:
        #                     if p_pointer < len(tmp_p):
        #                         p_pointer += 1
        #                         if p_pointer < len(tmp_p):
        #                             p_j_now, m_p_now_ij = tmp_p[p_pointer]
        #                         else:
        #                             p_j_now = float("inf")
        #             else:
        #                 q_j_now, m_q_now_ij = q[q_pointer]
        #                 break
        #             q_pointer += 1
        #         if (p_pointer == len(tmp_p) and q_pointer == len(tmp_p)) or q_j > C:
        #             break
        # return p[n][-1][1]
###################################################################################


### 金钱兑换问题：有一个货币系统，它有 n 种硬币，面值为 v_1,v_2, ... ,v_n，且 v_1 = 1
### 当兑换价值为 y 的钱时，求使得硬币最少的兑换方法

# 思路
# c[i,j] 表示价值为 j，有前 i 种硬币可选的情况下的最少方法数目
# we have recursive equality:
# c[i,j] = min(c[i-1, j], min(c[i-1, j-k*v_i]+k*v_j))

class MoneyExchane:

    def solve_by_2d_array(self, y:int, V:list):
        n = len(V)
        c = [[0 for i in range(y+1)] for j in range(n+1)]

        # initialize c
        for i in range(y+1):
            c[0][1] = 0
            c[1][i] = i
        for i in range(2, n+1):
            for j in range(1, y+1):
                value1 = c[i-1][j]
                value2 = 0x3f3f3f3f
                # max tmp for j-tmp*v_i >= 0
                tmp = j // V[i-1]
                if tmp:
                    value2 = min(
                        c[i-1][j-k*V[i-1]]+k for k in range(1, 1 + tmp)
                    )
                c[i][j] = min(value1, value2)

        return c[n][y]



class Floyd:

    def solve_by_2d_array(self, G):
        '''
        find min distance of each pairs

        :param G: the directed graph G
        :return: min distance of each pairs
        '''
        n = len(G)
        d = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                d[i][j] = G[i][j]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    d[i][j] = min(d[i][j], d[i][k]+d[k][j])
            # print(d)
        return d


#### 二叉搜索树：
##### 定义：
# 1. 每个节点作为搜索对象，关键字互不相同
# 2. 某节点左子树上所有节点关键字小于该节点的关键字
# 3. 某节点右子树上所有节点关键字大于该节点的关键字

# OBST(optimal binary search tree) 最优二叉搜索树问题：
# 给定一个 n 个不同关键字的已排序序列 K = <k_i> i \in [1,n]
# 对于每个关键字，都有一个概率 p_i 表示其搜索频率
# 扩充二叉树：还有 n+1 个伪关键字 d_0, d_1, ... , d_n 表示不在 K 中的值
# d_0 表示所有小于 k_1 的值，d_n 表示所有大于 k_n 的值，d_i 在 k_i 和 k_{i+1} 之间
# d_i 对应的概率为 q_i



class OBSTree:

    def find_min_cost_by_dp(self, p, q, K):
        '''
        find the optimal binary search tree

        :param p: the frequency of each key word
        :param q: the frequency of the not key word
        :param K: the sorted key word
        :return:
        '''

        assert sum(p[1:]) + sum(q) == 1
        assert len(p) == len(q)

        # the length of key word
        n = len(p)

        # w[i][j] indicate p_{i to j} + q_{i-1 to j}
        w = [[0 for i in range(n+1)] for j in range(n+1)]
        for i in range(1, n+1):
            for j in range(i-1, n):
                w[i][j] = sum(p[i:j+1]) + sum(q[i-1:j+1])

        # e[i][j] indicate the min cost when including key word k_{i to j}
        e = [[0 for i in range(n+1)] for j in range(n+1)]
        # initialize e
        for i in range(1, n+1):
            e[i][i-1] = q[i-1]

        # 2-d array root
        # root[i][j] indicate the root of p[i][j]
        root = [[0 for i in range(n+1)] for j in range(n+1)]


        # we have recursive equality
        # e[i][j] = min(e[i][r-1] + e[r+1][j] + w[i][j])
        for k in range(n-1):
            for i in range(1, n-k):
                cost = float("inf")
                index = -1
                for j in range(i, i+k+1):
                    now_cost = e[i][j-1] + e[j+1][i+k] + w[i][i+k]
                    if now_cost < cost:
                        cost = now_cost
                        index = j
                e[i][i+k] = cost
                root[i][i+k] = index

        # print tree in LDR order
        print("LDR order: ", end="")
        self.show_tree(root, K)
        print()
        return e[1][n-1]

    def show_tree(self, root, K):
        # LDR

        n = len(K)
        root_index = root[1][n] - 1
        if root_index < 0:
            return
        print(f"{K[root_index]}->", end="")
        self.show_tree(root, K[:root_index])
        self.show_tree(root, K[root_index+1:])


    @staticmethod
    def test():
        test = OBSTree()
        # 一个测试样例 期望搜索代价为 1.85
        test1_p = ['#', 0.5, 0.1, 0.05]
        test1_q = [0.15, 0.1, 0.05, 0.05]
        print(f"min cost: {test.find_min_cost_by_dp(test1_p, test1_q, [1, 2, 3])}")
        test2_p = ['#', 0.15, 0.10, 0.05, 0.10, 0.20]
        test2_q = [0.05, 0.10, 0.05, 0.05, 0.05, 0.10]
        print(f"min cost: {test.find_min_cost_by_dp(test2_p, test2_q, [1, 2, 3, 4, 5])}")


# 黑白点匹配问题
# 平面有 n 个白点和 n 个黑点，每一个点用坐标对 (x, y) 表示，
# 当且仅当黑点的横、纵坐标均不小于白点时，两点可能匹配
# 某一个黑点最多和一个白点匹配，某一个白点最多和一个黑点匹配
# 请找出一个最大的匹配



if __name__ == '__main__':
    # print("0-1 backpack problem")
    sol = ZeroOneKnapsack()
    # print(sol.solve_by_2d_array(10, [2,2,6,5,4], [6,3,5,4,6], 5))
    # print(sol.solve_by_1d_array(10, [2,2,6,5,4], [6,3,5,4,6], 5))
    print(sol.solve_by_2d_array(22, [3,5,7,8,9], [4,6,7,9,10], 5))
    # print(sol.solve_by_1d_array(200, [79, 58, 86, 11, 28, 62, 15, 68], [83, 14, 54, 79, 72, 52, 48, 62], 8))
    # print(sol.solve_by_2d_array(300, [95, 75, 23, 73, 50, 22, 6, 57, 89, 98], [89, 59, 19, 43, 100, 72, 44, 16, 7, 64], 10))
    # print(sol.solve_by_1d_array(300, [95, 75, 23, 73, 50, 22, 6, 57, 89, 98], [89, 59, 19, 43, 100, 72, 44, 16, 7, 64], 10))
    # print(sol.solve_by_2d_array(1000, [88, 85, 59, 100, 94, 64, 79, 75, 18, 38, 47, 11, 56, 12, 96, 54, 23, 6, 19, 31, 30, 32, 21, 31, 4, 30, 3, 12, 21, 60, 42, 42, 78, 6, 72, 25, 96, 21, 77, 36, 42, 20, 7, 46, 19, 24, 95, 3, 93, 73, 62, 91, 100, 58, 57, 3, 32, 5, 57, 50, 3, 88, 67, 97, 24, 37, 41, 36, 98, 52, 75, 7, 57, 23, 55, 93, 4, 17, 5, 13, 46, 48, 28, 24, 70, 85, 48, 48, 55, 93, 6, 8, 12, 50, 95, 66, 92, 25, 80, 16], [53, 70, 20, 41, 12, 71, 37, 87, 51, 64, 63, 50, 73, 83, 75, 60, 96, 70, 76, 25, 27, 89, 93, 40, 41, 89, 93, 46, 16, 4, 41, 29, 99, 82, 42, 14, 69, 75, 20, 20, 56, 23, 92, 71, 70, 1, 63, 18, 11, 68, 33, 6, 82, 69, 78, 48, 95, 42, 53, 99, 15, 76, 64, 39, 48, 83, 21, 75, 49, 73, 85, 28, 31, 86, 63, 12, 71, 35, 21, 17, 73, 18, 7, 51, 94, 88, 46, 77, 80, 95, 31, 80, 32, 45, 5, 30, 51, 63, 43, 9], 100))
    # print(sol.solve_by_1d_array(1000, [88, 85, 59, 100, 94, 64, 79, 75, 18, 38, 47, 11, 56, 12, 96, 54, 23, 6, 19, 31, 30, 32, 21, 31, 4, 30, 3, 12, 21, 60, 42, 42, 78, 6, 72, 25, 96, 21, 77, 36, 42, 20, 7, 46, 19, 24, 95, 3, 93, 73, 62, 91, 100, 58, 57, 3, 32, 5, 57, 50, 3, 88, 67, 97, 24, 37, 41, 36, 98, 52, 75, 7, 57, 23, 55, 93, 4, 17, 5, 13, 46, 48, 28, 24, 70, 85, 48, 48, 55, 93, 6, 8, 12, 50, 95, 66, 92, 25, 80, 16], [53, 70, 20, 41, 12, 71, 37, 87, 51, 64, 63, 50, 73, 83, 75, 60, 96, 70, 76, 25, 27, 89, 93, 40, 41, 89, 93, 46, 16, 4, 41, 29, 99, 82, 42, 14, 69, 75, 20, 20, 56, 23, 92, 71, 70, 1, 63, 18, 11, 68, 33, 6, 82, 69, 78, 48, 95, 42, 53, 99, 15, 76, 64, 39, 48, 83, 21, 75, 49, 73, 85, 28, 31, 86, 63, 12, 71, 35, 21, 17, 73, 18, 7, 51, 94, 88, 46, 77, 80, 95, 31, 80, 32, 45, 5, 30, 51, 63, 43, 9], 100))
    # print(sol.solve_by_2d_array(22, [3, 5, 7, 8, 9], [4, 6, 7, 9, 10], 5))
    # print(sol.solve_by_1d_array(22, [3, 5, 7, 8, 9], [4, 6, 7, 9, 10], 5))
    # print("="*100)
    # print("floyd algorithm")
    # floyd = Floyd()
    # print(floyd.solve_by_2d_array([[0,1,0x3f3f3f3f,2],[2,0,0x3f3f3f3f,2],[0x3f3f3f3f,9,0,4], [8,2,3,0]]))
    # print("="*100)
    # print("optimal binary search tree")
    # OBSTree.test()
    # money = MoneyExchane()
    # print(money.solve_by_2d_array(12, [1, 13]))