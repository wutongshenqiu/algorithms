### 第一章的算法题


# T1.35 设计一个时间复杂度为 O(1) 的算法，他在 A 中找出一个既不是最大值也不是最小值的元素
# 思路，只需要找出三个不同的元素即可，在这三个不同的元素中找出既不是最大也不是最小的
# 事实上可能并不存在一个时间复杂度为 O(1) 的算法，因为这样的数可能不存在，例如 A 中的数全部相等
def q1_35(A):
    a = A[0]
    for temp in A:
        if temp != a:
            b = temp
            break
    else:
        raise Exception("can't find target value!")
    for temp in A:
        if temp != a and temp != b:
            c = temp
            break
    else:
        raise Exception("can't find target value!")

    if a < b < c or c < b < a:
        return b
    if a < c < b or b < c < a:
        return c
    if c < a < b or b < a < c:
        return a

assert q1_35([1, 2, 3]) == 2
assert q1_35([1, 1, 2, 3, 4]) not in (1, 4)


