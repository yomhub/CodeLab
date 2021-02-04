from bisect import bisect_right

def treat(sells,buys):
    ans = -1
    ids,idb=0,0
    while(ids<len(sells)):
        hit=False
        for o in buys:
            if(o[1]>0 and o[0]>=sells[ids][0]):
                # deal
                ans=sells[ids][0]
                hit = True
                if(o[1]>=sells[ids][1]):
                    o[1]-=sells[ids][1]
                    sells[ids][1]=0
                    sells[ids][0]=0
                else:
                    sells[ids][1]-=o[1]
                    o[1]=0
        ids+=1

    return ans

def solution(stocks):
    # stocks['name'][0-Sell,1-Buy][0-Price,1-Num]
    deals={}
    for sname in stocks:
        ret = treat(stocks[sname][0],stocks[sname][1])
        if(ret>0):
            deals[sname]=ret
    if(len(deals)):
        for sname in deals:
            print('{}:{}'.format(sname,deals[sname]))
    else:
        print('Stocks not traded')

if __name__ == "__main__":
    n = int(input())
    stocks={}
    for i in range(n):
        cs = input()
        cs=cs.split(' ')
        if(cs[2] not in stocks):
            stocks[cs[2]]=[[],[]]
        if(cs[1]=='Sell'):
            stocks[cs[2]][0].append([int(cs[-2]),int(cs[-1])])
        else:
            stocks[cs[2]][1].append([int(cs[-2]),int(cs[-1])])

    solution(stocks)
    pass