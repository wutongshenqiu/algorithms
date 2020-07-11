def perm(A, n, k):
    if k == n:
        print(A)
    else:
        for i in range(k, n):
            A[k], A[i] = A[i], A[k]
            perm(A, n, k+1)
            A[k], A[i] = A[i], A[k]


def quick_sort(A, low, high):
    if low < high:
        w = A[low]
        i = low
        j = high
        while i < j:
            while A[j] >= w and i < j:
                j -= 1
            A[i] = A[j]
            while A[i] <= w and i < j:
                i += 1
            A[j] = A[i]
        A[i] = w
        quick_sort(A, low, i-1)
        quick_sort(A, i+1, high)

if __name__ == '__main__':
    # perm([1, 2, 3], 3, 0)
    import random
    import time
    import sys
    sys.setrecursionlimit(1000000)
    # this is a very bad performance
    A = list(range(3900))
    start = time.time()
    quick_sort(A, 0, len(A)-1)
    end = time.time()
    print(f"{end-start:.2f}s")

