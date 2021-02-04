import sys
import os
import numpy as np


def find_all_0_sub_mtx(mtx):
    """
      Find all sub matrix which had 0 sum.
        mtx: ndarray of martix.
      Return: number of total 0 sum sub matrix.
    """
    ret = 0

    def _sub_mtx(mtx):
        ret = 0
        if(max(mtx.shape[0], mtx.shape[1]) > 2):
            ax0 = mtx.shape[0]
            ax1 = mtx.shape[1]
            if(ax0 > 2):
                for i in range(ax0-1):
                    ret += _sub_mtx(mtx[i:i+2, :])
            if(ax1 > 2):
                for i in range(ax1-1):
                    ret += _sub_mtx(mtx[:, i:i+2])
        elif(max(mtx.shape[0], mtx.shape[1]) == 1):
            return 0
        if(np.sum(mtx) == 0):
            ret += 1
        return ret
    ret = _sub_mtx(mtx)
    return ret


def numIslands(grid: list) -> int:
    if(not grid or not grid[0]):
        return 0
    b_accessed = [[False for j in range(len(grid[0]))]
                  for i in range(len(grid))]
    len_x, len_y = len(grid[0]), len(grid)
    s_queue = []
    # BFS
    ans = 0
    is_iso = False
    for dy, by in enumerate(b_accessed):
        for dx, bx in enumerate(by):
            if(b_accessed[dy][dx]):
                continue
            b_accessed[dy][dx] = True
            if(grid[dy][dx]=='1'):
                s_queue.append((dy, dx))
                while(s_queue):
                    c_node = s_queue.pop(0)
                    b_accessed[c_node[0]][c_node[1]]=True
                    if(c_node[1]>0 and b_accessed[c_node[0]][c_node[1]-1]==False and grid[c_node[0]][c_node[1]-1]=='1'):
                        s_queue.append((c_node[0],c_node[1]-1))
                    if(c_node[0]>0 and b_accessed[c_node[0]-1][c_node[1]]==False and grid[c_node[0]-1][c_node[1]]=='1'):
                        s_queue.append((c_node[0]-1,c_node[1]))
                    if(c_node[1]<len_x-1 and b_accessed[c_node[0]][c_node[1]+1]==False and grid[c_node[0]][c_node[1]+1]=='1'):
                        s_queue.append((c_node[0],c_node[1]+1))
                    if(c_node[0]<len_y-1 and b_accessed[c_node[0]+1][c_node[1]]==False and grid[c_node[0]+1][c_node[1]]=='1'):
                        s_queue.append((c_node[0]+1,c_node[1]))
                ans += 1
    return ans


def combinationSum1(candidates:list, target:int):
    """
    任意长子数组凑target
    可重复使用成员
    """
    candidates = sorted(candidates) # sort to terminate early when target < 0
    
    def backtracking(i, target, path):
        if target == 0:
            ans.append(path)
            return
        if i == len(candidates) or target < candidates[i]:
            return
        backtracking(i, target - candidates[i], path + [candidates[i]]) # Choose ith candidate
        backtracking(i + 1, target, path) # Skip ith candidate
    
    ans = []
    backtracking(0, target, [])
    return ans

from collections import Counter
def nonDivisibleSubset(k, s):
    # Write your code here
    s = [o%k for o in s]
    cntdict = Counter(s)
    ans = 0
    for o1 in cntdict:
        tmp = cntdict[o1]
        for o2 in cntdict:
            if(o2==0 or o2==o1):
                continue
            elif(o2+o1!=k):
                tmp=max(tmp,cntdict[o2]+cntdict[o1])
        ans = max(ans,tmp)
    if(0 in cntdict and cntdict[0]==1):
        ans+=1
    return ans

if __name__ == "__main__":
    tt = (
        7,
        [278, 576, 496, 727, 410, 124, 338, 149, 209, 702, 282, 718, 771, 575, 436]
    )
    print(nonDivisibleSubset(*tt))
    pass
