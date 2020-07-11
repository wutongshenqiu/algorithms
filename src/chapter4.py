### 对应第4章的算法题目
from collections.abc import Iterable


class Heap:

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
            if self.seq[i] <= self.seq[i//2]:
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
            if j + 1 < len(self.seq) and self.seq[j+1] > self.seq[j]:
                j = j + 1
            if self.seq[i] >= self.seq[j]:
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
        if last >= tmp:
            self.shift_up(index)
        else:
            self.shift_down(index)

    def make_heap(self):
        # for i in range((len(self.seq)-1)//2, 0, -1):
        #     self.shift_down(i)
        for i in range((len(self.seq)-1)//2, len(self.seq)):
            self.shift_up(i)

    # 对应 T4.19
    # 合并两个堆，相当于重新构造，时间复杂度为 O(n)
    def merge(self, h):
        if not isinstance(h, Heap):
            raise Exception("must be a heap!")
        return Heap(self.seq[1:] + h.seq[1:])

    # 对应 T4.5
    # 判断一个序列是否是堆
    # 采用递归的方法，即中间节点大于孩子节点，时间复杂度为 O(n)
    # seq 表示序列，i 表示当前节点的序号
    def jduge_heap(self, seq, i=1, index=0):
        # 置标志位
        self.jduge_flag = True
        if index == 0:
            new_seq = ['#']
            new_seq.extend(seq)
            self._jduge_heap_detail(new_seq, i)
        else:
            self._jduge_heap_detail(seq, i)
        return self.jduge_flag

    def _jduge_heap_detail(self, seq, i):
        if not self.jduge_flag:
            return
        if 2 * i >= len(seq):
            return
        if 2 * i + 1 >= len(seq):
            if seq[i] >= seq[2*i]:
                self._jduge_heap_detail(seq, 2*i)
            else:
                self.jduge_flag = False
        else:
            if seq[i] >= max(seq[2*i], seq[2*i+1]):
                self._jduge_heap_detail(seq, 2*i)
                self._jduge_heap_detail(seq, 2*i+1)
            else:
                self.jduge_flag = False

    # 以下内容均为测试样例
    def test_4_5(self):
        assert self.jduge_heap([8, 6, 4, 3, 2])
        assert self.jduge_heap([7])
        assert self.jduge_heap([9, 7, 5, 6, 3])
        assert self.jduge_heap([9, 4, 8, 3, 2, 5, 7])
        assert not self.jduge_heap([9, 4, 7, 2, 1, 6, 5, 3])

    def test_shift_up(self):
        test1 = Heap([1])
        test1.shift_up(1)
        assert test1.seq == ["#", 1]
        test2 = Heap([1, 3, 5])
        test2.shift_up(2)
        assert test2.seq == ["#", 3, 1, 5]
        test2.shift_up(3)
        assert test2.seq == ["#", 5, 1, 3]

    def test_shift_down(self):
        test1 = Heap([1])
        test1.shift_down(1)
        assert test1.seq == ["#", 1]
        test2 = Heap([1, 3, 5])
        test2.shift_down(1)
        assert test2.seq == ["#", 5, 3, 1]
        test2.shift_down(2)
        assert test2.seq == ["#", 5, 3, 1]

    def test_insert_and_delete(self):
        test1 = Heap([8, 6, 4, 3, 2])
        test1.insert(100)
        assert self.jduge_heap(test1.seq, index=1)
        test1.delete(1)
        assert self.jduge_heap(test1.seq, index=1)

    def test_make_heap(self):
        test1 = Heap([1, 2, 3, 4, 12, 32, 2])
        assert not self.jduge_heap(test1.seq, index=1)
        test1.make_heap()
        assert self.jduge_heap(test1.seq, index=1)

    def test_merge(self):
        test1 = Heap([1, 2, 3, 4, 5])
        test1.make_heap()
        test2 = Heap([2, 3, 4, 5, 6])
        test2.make_heap()
        assert self.jduge_heap(test1.seq, index=1)
        assert self.jduge_heap(test2.seq, index=1)
        h = test1.merge(test2)
        assert isinstance(h, Heap)
        h.make_heap()
        assert self.jduge_heap(h.seq, index=1)

    def test_all(self):
        self.test_4_5()
        self.test_shift_up()
        self.test_shift_down()
        self.test_insert_and_delete()
        self.test_make_heap()
        self.test_merge()



# 不相交集的相关内容

class DisjointSet:

    def __init__(self, n):
        # 初始化
        self.seq = ["#"]
        # 格式为 (parent, rank)
        self.seq.extend([[-1, 1] for i in range(n)])
        for i in range(1, n):
            self.union(i, i+1)

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

    def test_union(self):
        test1 = DisjointSet(16)
        for i in range(1, 17):
            assert test1.seq[i][1] <= 4



if __name__ == "__main__":
    # h = Heap([1])
    # h.test_all()
    # d = DisjointSet(1)
    # d.test_union()
    import time
    a = list(range(100000000))
    start = time.time()
    sorted(a, reverse=True)
    end = time.time()
    print(f"{end-start:.2f}s")