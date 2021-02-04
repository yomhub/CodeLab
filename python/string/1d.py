import collections
import sys
import os
import functools
from collections import Counter

def strongPasswordChecker(s: str) -> int:
    ret = 0
    if(len(s) > 20):
        ret += len(s)-20
    elif(len(s) < 6):
        ret = 6-len(s)
        if(len(s) <= 3):
            return ret

    f_digt = False
    f_upc = False
    f_loc = False
    cc = 0
    tmp_cont = 0
    tmm_chr = ''
    should_del = 0

    for i in range(len(s)):
        if(f_loc or s[i].islower()):
            f_loc = True
        if(f_upc or s[i].isupper()):
            f_upc = True
        if(f_digt or s[i].isdigit()):
            f_digt = True
        if(tmm_chr == s[i]):
            tmp_cont += 1
            if(tmp_cont % 3 == 0):
                tmp_cont = 0
                cc += 1
        else:
            tmm_chr = s[i]
            if(tmp_cont >= 3):
                should_del += tmp_cont-2
            tmp_cont = 1

    tmp_cont = 0
    if(not f_digt):
        tmp_cont += 1
    if(not f_upc):
        tmp_cont += 1
    if(not f_loc):
        tmp_cont += 1

    if(len(s) < 6):
        return max(ret, cc, tmp_cont)
    elif(len(s) <= 20):
        return max(cc, tmp_cont)
    else:
        return max(ret+tmp_cont, ret+cc)


def isValidHTML(code: str) -> bool:
    """
    格式匹配：<大写或下划线></大写或下划线>
    """
    def _valid_tag_name(subcode: str):
        if(len(subcode) > 9):
            return False
        for o in subcode:
            if(not (('A' <= o and 'Z' >= o) or o == '_' or o == '/')):
                return False
        return True

    if(code[0] != '<' or code[-1] != '>' or len(code) <= 5):
        return False
    st = 0
    pe_code = len(code)
    start_tag_list = []

    # ensure 1st <> is valid tag
    et = code.find('>', st, pe_code)
    if(_valid_tag_name(code[1:et])):
        start_tag_list.append(code[st+1:et])
    else:
        return False
    ps_code = et+1

    while(True):
        st = code.find('<![CDATA[', ps_code, pe_code)
        if(st == -1):
            break
        et = code.rfind(']]>', st+7, pe_code)
        if(et == -1 or et+3 >= len(code)):
            return False
        code = code[0:st]+code[et+3:]
        pe_code = len(code)

    while(ps_code < len(code)):

        st = code.find('<', ps_code, pe_code)
        if(st == -1):
            return False
        et = code.find('>', st, pe_code)
        if(et == -1):
            return False

        if(et-st-1 > 0):
            if(code[st+1] != '/'):
                start_tag_list.append(code[st+1:et])
            else:
                if(len(start_tag_list) == 0):
                    return False
                if(code[st+2:et] != start_tag_list.pop()):
                    return False
                else:
                    if(not _valid_tag_name(code[st+2:et])):
                        return False
                    if(len(start_tag_list) == 0 and (et+2) < len(code)):
                        return False

        ps_code = et+1

    return len(start_tag_list) == 0


def isValidHTML_re(code: str) -> bool:
    import re
    # \代表反转义：\[ == """["""
    code = re.sub('\<\!\[CDATA\[.*?\]\]\>', 'c', code)
    while(re.search(r'\<([A-Z]{1,9})\>([^<]*)\<\/(\1)\>', code) != None):
        code = re.sub(r'\<([A-Z]{1,9})\>([^<]*)\<\/(\1)\>', 't', code)
    return code == 't'


def numDecodings(s: str) -> int:
    def _2ch(s: str) -> int:
        if(s == '**'):
            return 9*9+9+6
        elif(s[0] == '*'):
            # *d
            return 9+2
        elif(s[1] == '*'):
            # d*
            if(s[0] == '1'):
                return 18
            elif(s[0] == '2'):
                return 15
            return 9
        else:  # dd
            return 2 if(s[0] in ['1', '2'])else 1

    @functools.lru_cache(None)
    def _subdec(s: str) -> int:
        if(len(s) == 1):
            return 9 if(s[0] == '*')else 1
        elif(len(s) == 2):
            return _2ch(s)

        if(s[0] in ['1', '2']):
            return _subdec(s[1:]) + _2ch(s[0:2])*_subdec(s[2:])
        elif(s[0] == '*'):
            return 9*_subdec(s[1:]) + _2ch(s[0:2])*_subdec(s[2:])
        else:
            return _subdec(s[1:])

    return _subdec(s) % (10**9+7)


def hcf(x, y):
    if(x > y):
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller + 1):
        if((x % i == 0) and (y % i == 0)):
            ret = i
    return ret


def repeatedSubstringPattern(s: str) -> bool:
    dit = {}
    for o in s:
        if(o not in dit):
            dit[o] = 1
        else:
            dit[o] += 1
    cur = None
    for name in dit:
        if(cur == None):
            cur = (name, dit[name])
        elif(cur[1] != dit[name]):
            di = hcf(cur[1], dit[name])
            if(di == 1):
                return False
            if(cur[1] > dit[name]):
                cur = (name, dit[name])
    tmp = s.split(cur[0])

    for i in range(len(tmp)):
        if(not tmp[i]):
            tmp.pop(i)
    max_len = len(tmp)//2

    for j in range(1, max_len+1):
        if(len(tmp) % j != 0):
            continue
        tmp_str = ''
        for o in tmp[0:j]:
            tmp_str += o
        i = j
        while(i < len(tmp)):
            tmp_str2 = ''
            for o in tmp[i:i+j]:
                tmp_str2 += o
            if(tmp_str2 != tmp_str):
                break
            i += j
        if(i >= len(tmp)):
            return True

    return False


def makeLargestSpecial(S):
    i = 0  # start of the substring
    count = 0  # to find a valid substring by counting 1's and 0's
    res = []  # store the transformed substrings
    for j, v in enumerate(S):
        # j would be the end of the substring
        # v is the current character being checked
        if v == "1":
            count += 1
        else:
            count -= 1
        # add to count when seeing 1, reduce count when seeing 0
        # when count is 0 again, means we just parsed a valid substring
        if count == 0:
            # take the inside of the substring, transform it using recursion
            # and add to the set of completed results
            res.append('1' + makeLargestSpecial(S[i + 1:j]) + '0')
            # move the start of the substring to the next character
            i = j + 1
    res.sort()  # make sure the substrings are ordered from lexigraphically smallest to largest
    res = res[::-1]  # now it's largest to smallest
    # join and return
    return ''.join(res)


def isMatch(s: str, p: str) -> bool:
    import re
    while(p.find('**') != -1):
        p = p.replace('**', '*')
    p = p.replace('?', '.').replace('*', '[a-z]*')
    p = re.compile(p)
    return p.fullmatch(s) != None


class RemoveInvalidParentheses:
    """
    输入一个字符串，输出所有可能的最小改动的括号匹配的字符串列表
    "))("->[""]
    "())()"->["()()"]
    "()())()"->["()()()", "(())()"]
    "(a)())()"->["(a)()()", "(a())()"]
    使用DFS递归实现
    """

    def removeInvalidParentheses(self, s: str) -> list:
        self.ret = []
        l, r = self.isValid(s)
        if(len(s) == 0):
            return ['']
        elif(l == 0 and r == 0):
            return [s]

        self.subSet(s, l, r, 0)
        return self.ret

    def isValid(self, s: str) -> bool:
        # 为避免)(合法，需要用两个计数器
        l, r = 0, 0
        for o in s:
            if(o == '('):
                l += 1
            elif(o == ')'):
                if(l > 0):
                    l -= 1
                else:
                    r += 1
        return l, r

    def subSet(self, s: str, num_l: int, num_r: int, ind: int):
        if(num_l == 0 and num_r == 0 and self.isValid(s) == (0, 0)):
            self.ret.append(s)
            return
        for i in range(ind, len(s)):
            if(i > 0 and s[i] == s[i-1]):
                continue
            if(s[i] == '(' and num_l > 0):
                # 只查找i位之后的
                self.subSet(s[:i]+s[i+1:], num_l-1, num_r, i)
            elif(s[i] == ')' and num_r > 0):
                self.subSet(s[:i]+s[i+1:], num_l, num_r-1, i)
        return


class FreedomTrailGolbal:
    """
    给定一个环ring: str，每次可以顺/逆时针转动一位，转动一个位要一个周期
    给定key: str使全部key在ring: str
    有个读取器位于ring[0]可以读取当前位，读取要一个周期
    求最短的周期使得环可以拼凑出key: str
    一维TSP的全局解法
    """

    def findRotateSteps(self, ring: str, key: str) -> int:
        self.ring = ring
        # 计算对于特定target，ring有几个能去的位置
        ring_dit = collections.Counter(ring)
        # 分支，(当前周期，当前位置)，读取要一个周期所以初始为len(key)
        bunchs = [[len(key), 0]]

        for target in key:
            if(ring_dit[target] == 1):
                # 如果只有一个可以去的，直接转过去
                for i in range(len(bunchs)):
                    tmp = self.cal1WayDet(target, bunchs[i][1], bunchs[i][0])
                    bunchs[i] = tmp
            else:
                # 如果只有多个可以去的，将所有可能转换为分支
                new_bunch = []
                for o in bunchs:
                    new_bunch += self.calNWayDet(target, o[1], o[0])
                bunchs = new_bunch
        return min(bunchs, key=lambda x: x[0])[0]

    def calNWayDet(self, target: str, start_i: int, baseDet: int) -> list:
        ret = []
        for i, o in enumerate(self.ring):
            if(o == target[0]):
                if(i == start_i):
                    ret.append((baseDet, i))
                elif(i < start_i):
                    ret.append(
                        (min(start_i-i, len(self.ring)-start_i+i)+baseDet, i))
                else:
                    ret.append(
                        (min(i-start_i, len(self.ring)-i+start_i)+baseDet, i))
        return ret

    def cal1WayDet(self, target: str, start_i: int, baseDet: int):
        for i, o in enumerate(self.ring):
            if(o == target[0]):
                if(i == start_i):
                    return baseDet, i
                elif(i < start_i):
                    return min(start_i-i, len(self.ring)-start_i+i)+baseDet, i
                else:
                    return min(i-start_i, len(self.ring)-i+start_i)+baseDet, i
                break

        return baseDet, i


class FreedomTrailDP:
    """
    给定一个环ring: str，每次可以顺/逆时针转动一位，转动一个位要一个周期
    给定key: str使全部key在ring: str
    有个读取器位于ring[0]可以读取当前位，读取要一个周期
    求最短的周期使得环可以拼凑出key: str
    一维TSP的局部最优解
    """

    def findRotateSteps(self, ring: str, key: str) -> int:
        self.ring = ring
        ring_dit = collections.defaultdict(list)
        for i, o in enumerate(ring):
            ring_dit[o].append(i)

        if(len(key) == 0):
            return 0
        elif(len(key) == 1):
            return 1+min(ring_dit[key[0]])

        dp = [0 for i in range(len(self.ring))]
        for iTo in ring_dit[key[0]]:
            dp[iTo] = self.calDet(0, iTo)
        for i in range(len(key)-1):
            if(key[i] == key[i+1]):
                continue
            for iTo in ring_dit[key[i+1]]:
                dp[iTo] = min([self.calDet(iFrom, iTo)+dp[iFrom]
                               for iFrom in ring_dit[key[i]]])

        ret = min([dp[i] for i in ring_dit[key[-1]]])

        return ret+len(key)

    def calDet(self, iFrom: int, iTo: int) -> int:
        if(iTo < iFrom):
            return min(iFrom-iTo, len(self.ring)-iFrom+iTo)
        return min(iTo-iFrom, len(self.ring)-iTo+iFrom)


def getHint(secret: str, guess: str) -> str:
    a, b = 0, 0
    machs = [False]*len(secret)
    machg = [False]*len(guess)
    for i in range(len(guess)):
        if(guess[i] == secret[i]):
            a += 1
            machs[i] = True
            machg[i] = True
    for i in range(len(guess)):
        if(machg[i]):
            continue
        for j in range(len(guess)):
            if(j == i or machs[j]):
                continue
            elif(guess[i] == secret[j]):
                machs[j] = True
                machg[i] = True
                b += 1
                break
    return "{}A{}B".format(a, b)


def solve(n, d, x, y):
    """Returns the sizes of groups in descending order."""
    # Write your solution here.
    #
    # Warning: Printing unwanted or ill-formatted data to output will cause
    # the test cases to fail.
    n = min(100, max(n, 2))
    x = x[:100]
    y = y[:100]
    nodes = range(1, n+1)
    fnodes = [-1]*len(nodes)
    grps = []
    for i in range(len(x)):
        if(fnodes[x[i]-1] == -1 and fnodes[y[i]-1] == -1):
            if(x[i] != y[i]):
                grps.append([x[i], y[i]])
                fnodes[x[i]-1] = len(grps)-1
                fnodes[y[i]-1] = len(grps)-1
            else:
                grps.append([x[i]])
                fnodes[x[i]-1] = len(grps)-1
            continue
        elif(fnodes[x[i]-1] != -1 and fnodes[y[i]-1] != -1):
            if(fnodes[x[i]-1] == fnodes[y[i]-1]):
                continue
            grps[fnodes[x[i]-1]] += grps[fnodes[y[i]-1]]
            ind = fnodes[y[i]-1]
            grps.pop(ind)
            for j in range(len(fnodes)):
                if(fnodes[j] == ind):
                    fnodes[j] = fnodes[x[i]-1]
            continue
        elif(fnodes[x[i]-1] != -1):
            # fnodes[y[i]-1]==-1
            grps[fnodes[x[i]-1]].append(y[i])
            fnodes[y[i]-1] = fnodes[x[i]-1]
        elif(fnodes[y[i]-1] != -1):
            # fnodes[x[i]-1]==-1
            grps[fnodes[y[i]-1]].append(x[i])
            fnodes[x[i]-1] = fnodes[y[i]-1]
    for i, o in enumerate(fnodes):
        if(o == -1):
            grps.append([nodes[i]])
    ans = [len(grp) for grp in grps]

    ans.sort(reverse=True)
    return ans

def morganAndString(a, b):
    a = list(a)
    b = list(b)
    ans=''
    while(a and b):
        a1 = a.pop(0)
        while(b and b[0]<=a1):
            ans+=b.pop(0)
        ans+=a1
            
    if(a):
        for o in a:
            ans+=o
    elif(b):
        for o in b:
            ans+=o  
    return ans  

def longestSubstring(s: str, k: int) -> int:
    if(k<=0):return 0
    elif(k==1):return len(s)
    
    cnts = Counter(s)
    removec = [o for o in cnts if(cnts[o]<k)]
    search = [s]
    while(removec):
        spl = removec.pop(0)
        tmp = []
        while(search):
            searchs = search.pop(0)
            if(len(searchs)<k):
                continue
            tmp += [o for o in searchs.split(spl) if(len(o)>=k)]
        search = tmp
    ans = 0
    for o in search:
        if(len(o)<k or len(o)<ans):
            continue
        loc_cnt = Counter(o)
        if(min(loc_cnt.values())<k):
            ans = max(longestSubstring(o,k),ans)
        else:
            ans = max(ans,len(o))
    return ans

if __name__ == "__main__":
    print(longestSubstring("bbaaacbd",3))

    pass
