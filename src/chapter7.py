### dynamic programming

from src.timeclock import clock

class Fibonacci:

    def fibonacci_by_recursion(self, n):
        if n == 1 or n == 2:
            return 1
        else:
            return self.fibonacci_by_recursion(n-1) + self.fibonacci_by_recursion(n-2)

    def fibonacci_by_dp(self, n):
        if n == 1 or n == 2:
            return 1
        a = 1
        b = 1
        for i in range(n-2):
            a, b = b, a + b
        return b

    @clock
    def test_fibonacci_by_recursion(self, n=50):
        return self.fibonacci_by_recursion(n)

    @clock
    def test_fibonacci_by_dp(self, n=50):
        return self.fibonacci_by_dp(n)

    @staticmethod
    def test_fibonacci():
        test = Fibonacci()
        assert test.test_fibonacci_by_dp(35) == test.test_fibonacci_by_recursion(35)

### 算法思想
# 动态规划算法与分治法类似，其基本思想也是将待求解的问题分解成若干个子问题
# 和分治法的区别在于：
# 1. 主要用于优化问题（求最优解）
# 2. 子问题并不独立，即子问题是可能重复的（重复的子问题不需要重复计算）
## 如果能够保存已经解决的子问题的答案，而在需要时找出已求得的答案，就可以避免大量重复的计算

### 动态规划的实质是分治和消除冗余
#### 其基本的步骤为：
# 1. 找出最优解的性值，并刻画其结构特征
# 2. 递归的定义最优值
# 3. 以自底向上的方式计算出最优值
# 4. 根据计算最优值时得到的信息，构造最优解


# 矩阵连乘问题
# 给定 n 个矩阵（A_1,A_2,...,A_n）其中相邻的矩阵之间是可乘的
# 如何确定计算矩阵连乘积的计算次序，使得依次次序计算矩阵连乘积需要的数乘次数最少

class MatrixOrder:

    def solve_by_dp(self, n, p):
        '''
        Solution:
            define A[i:j] as the smallest amount of computation A_i*A_{i+1}*A_{i+2}*...*A_j
            if divide A[i:j] to A[i:k] and A[k+1:j]
            then we have A[i:j] = min{A[i:k]+A[k+1:j]+p_{i-1}*p_i*p_{i+1}}
            the result will be indicated by A[1:n]

        Args:
        :param n: indicate the length of multiplications
        :param p: p_{i-1}, p_i means the row and column of matrix A_i
        :return: minimize the amount of calculation
        '''

        # initialize the array
        A = [[0 for i in range(n+5)] for j in range(n+5)]

        # r 表示两个数之间的间隔
        for r in range(1, n):
            for i in range(1, n+1-r):
                j = r + i
                A[i][j] = min([A[i][k]+A[k+1][j]+p[i-1]*p[k]*p[j] for k in range(i, j)])
        return A[1][n]



# 最长公共子序列问题
# 对于给定序列 X = {x_1, x_2, ... ,x_n}, Y = {y_1, ... , y_m} 求最长公共子序列

class MaxSubString:
    pass



if __name__ == "__main__":
    # Fibonacci.test_fibonacci()
    sol1 = MatrixOrder()
    print(sol1.solve_by_dp(4, [10, 20, 50, 1, 100]))