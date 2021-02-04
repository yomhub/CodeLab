import os
import sys


def arithmetic_ep(ep_slist):
    """
      Evaluate the value of an arithmetic expression in Reverse Polish Notation.
      Valid operators are +, -, *, /. Each operand may be an integer or another expression.
      Args: ep_slist: a string type list. e.g. ["2", "1", "+", "3", "*"]
      Return: value
    """
    ops = {'+': lambda a, b: a+b,
           '-': lambda a, b: a-b,
           '*': lambda a, b: a*b,
           '/': lambda a, b: a/b}
    if(not(ep_slist[-1] in ops)):
        return int(ep_slist[-1])
    else:
        return ops[ep_slist[-1]](arithmetic_ep(ep_slist[:-2]), int(ep_slist[-2]))


def isValidSerialization(preorder: str) -> bool:
    if(len(preorder) < 5):
        return preorder == '#'
    nodes = preorder.split(',')
    if(nodes[-1] != '#'):
        return False
    tmp_level = 0

    for o in nodes[:-1]:
        if(o != '#'):
            tmp_level += 1
        else:
            tmp_level -= 1
        if(tmp_level < 0):
            return False

    tmp_level = tmp_level
    return tmp_level == 0


def trap(h:list) -> int:
    """
    用堆来查积水
    给定一个数组，代表一个柱状图的高度
    求该柱状图能积多少水
        ----   ----
     ---|  |---|  |
     |  |  |   |  |
    ----------------->d
    如上图可积累一个单位的水
    通过堆表示谷，保存左翼最高和右翼最高
    Det_d=新右翼(i)-当前位置
    当前位置积累的水 = Det_d*(min(新右翼，左翼)-旧右翼(top = st.pop(-1)))
    """
    i = ans = 0
    st = []
    while(i<len(h)):
        while(st and h[i]>h[st[-1]]):
            top = st.pop(-1)
            if(not st):
                break
            dist_i = i-st[-1]-1
            bounded_h = min(h[i],h[st[-1]])-h[top]
            ans+=dist_i*bounded_h
        st.append(i)
        i+=1
    return ans

class Solution:
    def findSubstring(self, s: str, words: list) -> list:
        loc_list = [[] for _ in range(len(words))]
        wrd_lens = [len(w) for w in words]
        min_l = min(wrd_lens)
        for i in range(len(s)-min_l+1):
            for j in range(len(words)):
                if(s[i:i+wrd_lens[j]]==words[j]):
                    loc_list[j].append(i)
        ans = []
        for o in loc_list:
            if(not o):
                return []
        
        def find_path(sp:int,curp:int,sub_loc:list,sub_wrd:list):
            if(not sub_loc and sp not in ans):
                ans.append(sp)
                return None
            for i in range(len(sub_loc)):
                curp2 = curp+sub_wrd[i]
                for o in sub_loc[i]:
                    if(o==curp2):
                        find_path(sp,curp2,sub_loc[:i]+sub_loc[i+1:],sub_wrd[:i]+sub_wrd[i+1:])
            return None
        
        for i in range(len(loc_list)):
            for sp in loc_list[i]:
                find_path(sp,sp,loc_list[:i]+loc_list[i+1:],wrd_lens[:i]+wrd_lens[i+1:])
        return ans

if __name__ == "__main__":
    p = Solution()
    tt=(
        "wordgoodgoodgoodbestword",
        ["word","good","best","good"],
    )
    print(p.findSubstring(*tt))
    print('hello')
