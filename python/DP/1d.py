import sys
import os
import math


def dp_1d_finite_reg(src_pool, target):
    """
      Finite 1d regression, find the closest combination from src_pool.
      Args:
        src_pool: 1d list.
        target: regression target.
      Return:
        Difference value, combination list.
    """
    cdiff = -1
    if(len(src_pool) == 1):
        if(target > src_pool[0]):
            cdiff = target - src_pool[0]
            csubp = src_pool
    else:
        for i in range(len(src_pool)):
            if(target > src_pool[i]):
                dif, subp = dp_1d_finite_reg(
                    src_pool[0:i]+src_pool[i+1:], target-src_pool[i])
                if(cdiff == -1 or dif < cdiff):
                    cdiff = dif
                    csubp = [src_pool[i]]+subp
    if(cdiff == -1):
        cdiff = target
        csubp = []
    return cdiff, csubp


def primes_sequence(A, B, C, D):
    """
    Given three prime numbers A, B and C and an integer D.
    find the first(smallest) D integers which only have A, B, C or a combination of them as their prime factors.
    """
    x = 0
    y = 0
    z = 0
    k = D
    ans = [0]*(k+1)
    ans[0] = 1
    for i in range(1, k+1):
        temp = min(A*ans[x], B*ans[y], C*ans[z])
        ans[i] = temp
        if temp == A*ans[x]:
            x += 1
        if temp == B*ans[y]:
            y += 1
        if temp == C*ans[z]:
            z += 1
    return ans[1:]


def numFactoredBinaryTrees(A: list) -> int:
    """
    Given an array of unique integers, each integer is strictly greater than 1.
    We make a binary tree using these integers and each number may be used for any number of times.
    Each non-leaf node's value should be equal to the product of the values of it's children.
    Return how many binary trees can it make.
    E.g. A=[2,4,5,10], return 7
    [2],[4],[5],[10],[2,2,4],[10,5,2],[10,2,5]

    """
    dp = {}
    for i in sorted(A):
        dp[i] = sum(dp[j] * dp.get(i/j, 0) for j in dp if i%j == 0) + 1
    return sum(dp.values()) % (10**9 + 7)


def longestArithSeqLength(A: list) -> int:
    """
    返回最长等差数列长度
    记忆表使用(i,diff)
    """
    d={}
    res=-1
    for i in range(len(A)):
        for j in range(i):
            diff=A[i]-A[j]
            if (j,diff) in d:
                d[(i,diff)]=d[(j,diff)]+1
            else:
                d[(i,diff)]=2
            res=max(res,d[(i,diff)])
    return res

def numRollsToTarget(d: int, f: int, target: int) -> int:
    """
    从[1,f]中抽d次数字求和
    给定d=投掷次数，f=可能面值，target=目标求和
    使用带动态规划表的递归
    """
    dp={}
    def roll(d,t):
        if t>d*f or d<1 or t<1:
            dp[(d,t)]=0
        else:
            # 尽可能避免或减小 o in dp操作
            if (d,t) not in dp:
                res=0
                if d>1:
                    for i in range(1,f+1):
                        res+=roll(d-1,t-i)
                elif t<=f:
                    res+=1
                dp[(d,t)]=int(res%(1e9+7))
        return dp[(d,t)]
    roll(d,t)
    return dp[(d,t)]

class PartitionKSubsets:
    """
    测目标数组能不能划分成等和的3份
    使用递归做
    """
    def canPartitionKSubsets(self, nums: list, k: int) -> bool:
        traget_sum = sum(nums)
        if(traget_sum%k!=0):return False
        self.traget_sum = traget_sum//k
        nums.sort()
        self.nums = nums[::-1]

        return self.canSplit(0,len(nums),[False]*len(nums),0,k)

    def canSplit(self, sp:int, ep:int, f_nums:list, c_sum:int, k: int) -> bool:
        if(k == 1):
            return True
        if(c_sum==self.traget_sum):return self.canSplit(0,ep,f_nums,0,k-1)
        for i in range(sp,ep):
            if(self.nums[i]+c_sum>self.traget_sum or f_nums[i]):
                continue
            # 注意这里是尝试，所以未成功之后要undo
            f_nums[i]=True
            if(self.canSplit(i+1,ep,f_nums,c_sum+self.nums[i],k)):return True
            f_nums[i]=False
            
        return False
        
def isMatch(s: str, p: str) -> bool:
    """
    字符串匹配，'?'代表任意单个字符，'*'代表0个或多个任意字符
    建立DP表[len(p) + 1][len(s) + 1]，dp[0][0]为True

    对于*：
        [idp+1]为当前状态，[idp]为上一个字符的状态
        [idp]   =   [F,F,F,...T,T,......]
                     ---------^---------
        找到第一个T, [idp+1]的状态即接受之后的全部
        [idp+1]  =  [F,F,F,...T,T,T,T,T,]
                     ---------^---------
        分支情况的处理包含在后面的一串T中, 意思是视后面的所有情况为可能分支
    对于?:
        上一个状态的位移, [idp+1现状态][ids当前字符] = [idp上一个状态][ids-1上一个字符]
        [idp]   =     [F,F,F,...T,T,......]
        [idp+1]  =  [F,F,F,...T,T,......]

    对于单字匹配:
        在照搬的同时加上限制: "必须要和当前字符相等"
        [idp]   =   [F,F,F,...T,T,......]
        [idp+1]  =  [F,F,F,...T,T,......] and 当前p == 对应位的s
    
    """
    if(len(p)==0 or len(s)==0):return False
    elif(p==s):return True
    dp = [[False for o in range(len(s) + 1)] for o in range(len(p) + 1)]
    dp[0][0] = True
    for idp,o in enumerate(p):
        if(o=='*'):
            ids = 1
            while((not dp[idp][ids-1]) and (ids < len(s) + 1)): 
                ids+=1
            dp[idp+1][ids-1] = dp[idp][ids-1]
            while (ids < len(s)+1):
                dp[idp+1][ids] = True
                ids+=1
        elif(o=='?'):
            for ids in range(1,len(s)+1):
                dp[idp+1][ids] = dp[idp][ids-1]            
        else:
            for ids in range(1,len(s)+1):
                dp[idp+1][ids] = dp[idp][ids-1] and p[idp] == s[ids-1]

    return dp[-1][-1]



def palindromePartition(s: str, k: int) -> int:
    s=s.lower()
    dp = [[0 for i in range(len(s))] for j in range(len(s))]
    
    for slen in range(2,len(s)+1):
        for i in range(0,len(s)-slen):
            j=i+slen-1
            if(slen==2):
                dp[i][j] = 0 if(s[i]==s[j])else 1
            else:
                dp[i][j] = dp[i+1][j-1] + (0 if(s[i]==s[j])else 1)
    inf = [[-1 for i in range(len(s))] for j in range(k)]
    
from bisect import bisect_left,bisect_right
def numPermsDISequence(S: str) -> int:
    ans = 0
    n = len(S)
    cand = list(range(n+1))
    def rec(val:int,subs:str, subc:list):
        if(not subs):
            ans+=len(subc)
            return
        if(not subc):
            return
        if(subs[0]=='D'):
            subsubc = subc[:bisect_left(subc,val)]
        else:
            subsubc = subc[bisect_right(subc,val):]
        
        if(len(subs)==1):
            ans+=len(subsubc)
            return
        elif(not subsubc):
            return
        else:
            for i in range(len(subsubc)):
                rec(subsubc[i],subs[1:],subsubc[:i]+subsubc[i+1:])
    for j in range(len(cand)):
        rec(cand[j],S,cand[:j]+cand[j+1:])
    return ans%(10**9+7)
if __name__ == "__main__":
    print(numPermsDISequence('DID'))
    pass