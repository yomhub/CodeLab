import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_left
v_range = 1.8
sigma = 0.59
# sigma = 0.5
# v_range=1.6
dx = np.linspace(0,v_range*2,500)
Fg = lambda x,sigma:np.exp(-((x)**2/(2.0*sigma**2)))
ys = np.where(dx<=v_range,1,0)

ax = plt.subplot()
# plt.xlim((0, 2*v_range))
# plt.ylim((0, 1.1))

ygs = Fg(dx,sigma)
idx = np.where(ygs>0.3)
idx = np.max(idx[0])
ys = np.array([72.1,78.5,86.1,95.7,107.4])
xs = np.arange(1,6)
ax.plot(xs,ys,label='Authentication Time(ms)')
# ax.plot(dx,ys,linestyle='-',label='01 classification')
plt.legend()
plt.xlabel("Lenth of certificate chain")
# plt.fill_between(dx[:idx], ygs[:idx], 0,
#                  facecolor="blue", # The fill color
#                  color='blue',       # The outline color
#                  alpha=0.2)          # Transparency of the fill
# plt.text(dx[idx],ygs[idx],"({:.2f},{:.2f})".format(dx[idx],ygs[idx]))
# plt.text(dx[250],ygs[250]+0.02,"({:.2f},{:.2f})".format(dx[250],ygs[250]))
plt.show()