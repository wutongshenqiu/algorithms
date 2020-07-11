##### 对应第五章的习题

class Solution:


    # T5.5 设计一个递归算法，在有 n 个元素的序列 A[1...n] 中寻找元素 x
    def q5_5(self, A, x, first, end):
        if first > end:
            return None
        if A[first] == x:
            return first
        else:
            return self.q5_5(A, x, first+1, end)





    def test_q5_5(self):
        test1 = [1, 2, 3, 4, 5]
        assert self.q5_5(test1, 1, 0, len(test1)-1) == 0
        assert self.q5_5(test1, 2, 0, len(test1)-1) == 1
        assert self.q5_5(test1, 5, 0, len(test1)-1) == 4
        assert self.q5_5(test1, 6, 0, len(test1)-1) == None

    def test_all(self):
        method_list = dir(self)
        for method in method_list:
            if method.startswith("test") and method != "test_all":
                getattr(self, method)()


class SmallHeap:

    def __init__(self, seq=None):
        # 下标从 1 开始
        self.seq = ['#']
        self.seq.extend(seq)

    # 堆中元素的上移操作
    def shift_up(self, i):
        # 如果是根节点
        if i < 1 or i >= len(self.seq):
            raise Exception("Invalid index i!")
        if i == 1:
            return
        while i > 1:
            if self.seq[i] >= self.seq[i//2]:
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
            if j + 1 < len(self.seq) and self.seq[j+1] < self.seq[j]:
                j = j + 1
            if self.seq[i] <= self.seq[j]:
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
        if last <= tmp:
            self.shift_up(index)
        else:
            self.shift_down(index)

    def make_heap(self):
        for i in range((len(self.seq)-1)//2, 0, -1):
            self.shift_down(i)

    # 对应 T4.19
    # 合并两个堆，相当于重新构造，时间复杂度为 O(n)
    def merge(self, h):
        if not isinstance(h, SmallHeap):
            raise Exception("must be a heap!")
        return SmallHeap(self.seq[1:] + h.seq[1:])

    # 弹出堆顶的元素
    def pop(self):
        tmp = self.seq[1]
        self.delete(1)
        return tmp



# 寻找第 k 小的元素
# 思路1，采用堆结构（优先队列）
def find_kmin_by_heap(A, k):
    if k <= 0:
        raise Exception("invalid k")
    h = SmallHeap(A)
    h.make_heap()
    for i in range(k):
        tmp = h.pop()
    return tmp


# 选择排序
def selectsort(A):
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if A[j] < A[i]:
                A[j], A[i] = A[i], A[j]
    return A

# 思路2，采用书上的递归算法
def select(A, low, high, k):
    p = high - low
    if p < 44:
        return find_kmin_by_heap(A, k)
    q = p // 5
    # the array of middle num
    middle_num = []
    for i in range(0, q*5, 5):
        A[i:i+5] = selectsort(A[i:i+5])
        middle_num.append(A[i+2])
    # find q // 2 num in mid array
    mm = select(middle_num, 0, q, q //2)
    # save num that smaller than mm
    A1 = []
    # save num that equal to mm
    A2 = []
    # save num that bigger than mm
    A3 = []
    for i in range(p):
        tmp = A[i+low]
        if tmp < mm:
            A1.append(tmp)
        elif tmp == mm:
            A2.append(tmp)
        elif tmp > mm:
            A3.append(tmp)
    # if |A_1| >= k, k_min_num must in A_1
    if len(A1) >= k:
        return select(A1, 0, len(A1), k)
    elif len(A1) + len(A2) >= k:
        return mm
    elif len(A1) + len(A2) < k:
        return select(A3, 0, len(A3), k-len(A1)-len(A2))


# 想法
# 当 k << n 时采用堆比较快
# 当 k -> n / 2 时采用 select 排序



if __name__ == "__main__":
    import time
    test1 = [100000000-i for i in range(100000000)]

    start = time.time()
    assert find_kmin_by_heap(test1, 50000000) == 50000000
    middle = time.time()
    print("%.2f" % (middle - start))
    assert select(test1, 0, 100000000, 50000000) == 50000000
    end = time.time()
    print("%.2f" % (end-middle))

