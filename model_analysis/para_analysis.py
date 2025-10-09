##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from S import ModelS
import matplotlib as mpl
##
# path
cur_path = os.getcwd()
output_path = os.path.join(cur_path, 'Output')
if not os.path.exists(output_path):
    os.makedirs(output_path)

##
# Para K
model_set = {}
var = {}
k_interval = range(5, 11)
for k in k_interval:
    model_set[k] = ModelS(k=k)
    model_set[k].init()
    x, y = model_set[k].s()
    var[k] = [x, y]

mpl.rcParams['figure.dpi'] = 100
# plt.style.use('seaborn-dark')

fig, ax = plt.subplots()
for parameter in k_interval:
    ax.plot(var[parameter][0], var[parameter][1], label='k='+str(parameter) )
ax.set(xlim=(0, 800))
plt.legend()
fig.supxlabel('Driver $x$')
fig.supylabel('State $y$')
plt.title('Para K  $(C_0=1,C_1=e)$')
plt.savefig(os.path.join(output_path, 'Para K.png'))

data = []
index = []
for parameter in k_interval:
    d = var[parameter][0]
    d1 = var[parameter][1]
    index.append('k='+str(parameter))
    data.append(d)
    index.append('y')
    data.append(d1)
a = pd.DataFrame(data, index=index).T
a.to_csv(os.path.join(output_path, 'Para_K.csv'), index=False)
##
# Para C_0
model_set = {}
var = {}
c_0_interval = np.linspace(1, 2, 5)
for c_0 in c_0_interval:
    model_set[c_0] = ModelS(c_0=c_0, k=9)
    model_set[c_0].init()
    x, y = model_set[c_0].s()
    var[c_0] = [x, y]

fig, ax = plt.subplots()
for parameter in c_0_interval:
    ax.plot(var[parameter][0], var[parameter][1], label='$C_0$='+str(parameter) )
ax.set(xlim=(0, 300))
plt.legend()
fig.supxlabel('Driver $x$')
fig.supylabel('State $y$')
plt.title('Para $C_0  (C_1=e,K=9)$')
plt.savefig(os.path.join(output_path, 'Para C_0.png'))

data = []
index = []
for parameter in c_0_interval:
    data.append(var[parameter][0])
    data.append(var[parameter][1])
    index.append('$C_0$='+str(parameter))
    index.append('y')
a = pd.DataFrame(data, index=index).T
a.to_csv(os.path.join(output_path, 'Para_C_0.csv'), index=False)

##
# Para C_1
model_set = {}
var = {}
c_1_interval = np.linspace(np.exp(1), np.exp(1)+2, 5)
for c_1 in c_1_interval:
    model_set[c_1] = ModelS(c_1=c_1, k=9)
    model_set[c_1].init()
    x, y = model_set[c_1].s()
    var[c_1] = [x, y]

fig, ax = plt.subplots()
for parameter in c_1_interval:
    ax.plot(var[parameter][0], var[parameter][1], label='$C_1$='+str(parameter) )
ax.set(xlim=(0, 300))
plt.legend()
fig.supxlabel('Driver $x$')
fig.supylabel('State $y$')
plt.title('Para $C_1  (C_0=1,K=9)$')
plt.savefig(os.path.join(output_path, 'Para C_1.png'))
plt.show()

data = []
index = []
for parameter in c_1_interval:
    data.append(var[parameter][0])
    data.append(var[parameter][1])
    index.append('$C_1$=' + str(parameter))
    index.append('y')
a = pd.DataFrame(data, index=index).T
a.to_csv(os.path.join(output_path, 'Para_C_1.csv'), index=False)
##
# any para
model_set = {}
var = {}
model_set[2] = ModelS(c_0=10, c_1=27.18, k=90)
model_set[2].init()
for para in model_set:
    x, y = model_set[para].s()
    var[para] = [x, y]

fig, ax = plt.subplots()
for parameter in model_set:
    ax.plot(var[parameter][0], var[parameter][1], label='$C_1$='+str(model_set[parameter].c_1)+',$C_0=$'+str(model_set[parameter].c_0)+',$K=$'+str(model_set[parameter].k))
ax.set(xlim=(0, 300))
plt.legend()
fig.supxlabel('Driver $x$')
fig.supylabel('State $y$')
plt.title('Para ')
plt.savefig(os.path.join(output_path, '10 Normal.png'))
plt.show()


##
# any para
model_set = {}
var = {}
model_set[2] = ModelS(k=9)
model_set[2].init()
for para in model_set:
    x, y = model_set[para].s()
    var[para] = [x, y]

fig, ax = plt.subplots()
for parameter in model_set:
    ax.plot(var[parameter][0], var[parameter][1], label='$C_1$='+str(model_set[parameter].c_1)+',$C_0=$'+str(model_set[parameter].c_0)+',$K=$'+str(model_set[parameter].k))
ax.set(xlim=(0, 300))
plt.legend()
fig.supxlabel('Driver $x$')
fig.supylabel('State $y$')
plt.title('Para normal')
plt.savefig(os.path.join(output_path, 'Normal.png'))
plt.show()
##
# Para C_0
model_set = {}
var = {}
c_0_interval = np.linspace(10, 15, 5)
for c_0 in c_0_interval:
    model_set[c_0] = ModelS(c_0=c_0, k=90, c_1=27.18)
    model_set[c_0].init()
    x, y = model_set[c_0].s()
    var[c_0] = [x, y]

fig, ax = plt.subplots()
for parameter in c_0_interval:
    ax.plot(var[parameter][0], var[parameter][1], label='$C_0$='+str(parameter) )
ax.set(xlim=(0, 300))
plt.legend()
fig.supxlabel('Driver $x$')
fig.supylabel('State $y$')
plt.title('Para $C_0  (C_1=10e,K=90)$')
plt.savefig(os.path.join(output_path, 'Para C_0 new.png'))