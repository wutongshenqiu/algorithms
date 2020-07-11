### homework3



# T1
# loc indicate the four corner
loc = [[i, j] for i in [-1, 0] for j in [-1, 0]]


class Solution:

    def __init__(self):
        self.flag = True

    def q_6_54(self, k, board):
        '''
        question 6.54 in page 128
        证明加强的结论：对于一个 n = 2^k 长度的棋盘，去掉该棋盘的四个角落的任意一个方格后
        剩下的方格恰好能被 L 型方格所覆盖
        :param k: the length of the checkerboard is 2^k
        :param board: a two dimension list that indicates the checkerboard,
        use 1 to imply occupy by L and 0 for not occupy
        :return: True or False
        '''

        # recursive export
        if not self.flag:
            return
        if k == 1:
            if board in [[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]]:
                return
            else:
                self.flag = False
                return

        # divide the question
        index = -1
        for i in range(4):
            tx, ty = loc[i]
            if board[tx][ty]:
                index = i
                break
        if index == -1:
            raise Exception("one corner must be 1!")
        else:
            mid = 1 << k-1
            tx, ty = loc[index]
            board[tx][ty] = 1
            # four small boards
            board1 = list(map(lambda x: x[:mid], board[:mid]))
            board2 = list(map(lambda x: x[mid:], board[:mid]))
            board3 = list(map(lambda x: x[:mid], board[mid:]))
            board4 = list(map(lambda x: x[mid:], board[mid:]))

            board1[mid-1][mid-1] = 1
            board2[mid-1][0] = 1
            board3[0][mid-1] = 1
            board4[0][0] = 1

            all_small_board = [board1, board2, board3, board4]

            # four situations
            if tx == 0 and ty == 0:
                board1[mid-1][mid-1] = 0
            if tx == 0 and ty == -1:
                board2[mid-1][0] = 0
            if tx == -1 and ty == 0:
                board3[0][mid-1] = 0
            if tx == -1 and ty == -1:
                board4[0][0] = 0

            for small_board in all_small_board:
                self.q_6_54(k-1, small_board)


            # recover the board
            board[tx][ty] = 0

    @staticmethod
    def test():
        sol1 = Solution()
        board1 = [[0 for i in range(1 << 1)] for j in range(1 << 1)]
        board1[0][0] = 1
        sol1.q_6_54(1, board1)
        assert sol1.flag
        sol2 = Solution()
        board2 = [[0 for i in range(1 << 2)] for j in range(1 << 2)]
        board2[-1][-1] = 1
        sol2.q_6_54(2, board2)
        assert sol2.flag
        sol3 = Solution()
        board3 = [[0 for i in range(1 << 3)] for j in range(1 << 3)]
        board3[0][-1] = 1
        sol3.q_6_54(3, board3)
        assert sol3.flag
        sol4 = Solution()
        board4 = [[0 for i in range(1 << 10)] for j in range(1 << 10)]
        board4[-1][0] = 1
        sol4.q_6_54(10, board4)
        assert sol4.flag






# T2 对寻找第 k 项的改进
# 1. 可行性是显然的
# 时间复杂度，最好情况 O(n)，最坏情况 O(kn)，平均情况 O(klogn)
# 2. 可行性是显然的，
# 时间复杂度，最好情况 O(n)，最坏情况 O(kn), 平均情况 O(klogn)

import random

class FindKthMin:

    # func determines how to choose the element that divide the array
    def division(self, A, low, high, k, func=random.choice):
        p = high - low
        mm = func(A)
        A1 = []
        A2 = []
        A3 = []
        for i in range(p):
            tmp = A[low+i]
            if tmp < mm:
                A1.append(tmp)
            if tmp == mm:
                A2.append(tmp)
            if tmp > mm:
                A3.append(tmp)
        if len(A1) >= k:
            return self.division(A1, low, len(A1), k)
        elif len(A1) + len(A2) >= k:
            return mm
        else:
            return self.division(A3, low, len(A3), k-len(A1)-len(A2))


    @staticmethod
    def test():
        test1 = FindKthMin()
        assert test1.division([1, 2, 3, 4], 0, 4, 2) == 2
        assert test1.division([i for i in range(100000)], 0, 100000, 50000) == 49999
        test2 = FindKthMin()
        func = lambda x: sum(x) / len(x)
        assert test2.division([1, 2, 3, 4, 4, 3, 2, 1], 0, 8, 3, func=func) == 2
        assert test2.division([i for i in range(100000)], 0, 100000, 50000, func=func) == 49999



if __name__ == '__main__':
    # Solution.test()
    FindKthMin.test()