class Calculator():
    def __init__(self):
        self.opt_def = [['*','/'],['+','-'],]
        self.fun_dict = {
            '+':lambda x,y:x+y,
            '-':lambda x,y:x-y,
            '*':lambda x,y:x*y,
            '/':lambda x,y:x/y,
        }
    def calculate(self,lamb:str):
        sp,ep = 0,0
        self.nums,self.opts = [[]],[[]]
        while(ep<len(lamb)):
            if(lamb[ep].isdigit() or lamb[ep]=='.'):
                ep+=1
                continue
            if(ep>sp):
                self.nums[-1].append(float(lamb[sp:ep]))
                sp = ep
                continue
            elif(lamb[sp]=="("):
                self.nums.append([])
                self.opts.append([])
            elif(lamb[sp]==")"):
                self.calc()
            else:
                self.opts[-1].append(lamb[sp])
            sp+=1
            ep=sp
        if(ep>sp):
            self.nums[-1].append(float(lamb[sp:ep]))
        self.calc()
        return self.nums[0][0]

    def calc(self):
        for opt_lv in self.opt_def:
            for opt in opt_lv:
                while(opt in self.opts[-1]):
                    idx = self.opts[-1].index(opt)
                    self.opts[-1].pop(idx)
                    x = self.nums[-1].pop(idx)
                    y = self.nums[-1].pop(idx)
                    self.nums[-1].insert(idx,self.fun_dict[opt](x,y))
        if(len(self.nums)>1):
            n = self.nums.pop(-1)[0]
            self.nums[-1].append(n)
            self.opts.pop(-1)

if __name__ == "__main__":
    obj = Calculator()
    ans = 15+8*(10+(50+8*9/3+3)/10)
    t = "15+8*(10+(50+8*9/3+3)/10)"
    # ans = (120)
    # t = '(120)'
    print(ans)
    print(obj.calculate(t))
    pass