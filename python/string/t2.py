def box_calculate(box):

    cnts = []
    maxv,maxid=1,[]
    for j in range(len(box[0])):
        tmp=0
        for i in range(len(box)):
            if(box[i][j]):
                tmp+=1
        cnts.append(tmp)
        if(maxv==tmp):
            maxid.append(j)
        elif(tmp>maxv):
            maxv=tmp
            maxid=[j]

    ans = 0
    for cj in maxid:
        detp = cj
        while(detp>0 and cnts[detp]>0):
            detp-=1
        ans = max(ans,abs(cj-detp))
        detp = cj
        while(detp<len(cnts) and cnts[detp]>0):
            detp+=1
        ans = max(ans,abs(cj-detp))

    return min(ans,maxv)

def solution(box):
    box_h = box
    box_v = [[False]*len(box) for _ in range(len(box[0]))]
    for i in range(len(box)):
        for j in range(len(box[0])):
            box_v[-1-j][i]=box[i][j]
    
    
    return max(box_calculate(box_h),box_calculate(box_v))
    
if __name__ == "__main__":
    n = int(input())
    box=[]
    for i in range(n):
        cs = input()
        cs=cs.split()
        box.append([True if(o=='C')else False for o in cs])
    print(solution(box))
    pass