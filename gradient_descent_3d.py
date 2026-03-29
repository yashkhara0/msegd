import numpy as np
import matplotlib.pyplot as plt

# initializing
X = np.linspace(1,10,10)
m = 3
c = 7
a = 0.2
Y = m*X + c
noisy = Y + np.random.normal(2,0.3,len(X))

# to find losses at each point for later plot and later terminator
def lossFind(m,c,X,y):
    loss = np.mean((y - (m*X+c))**2)
    return loss

# main func
def gradDesc(alpha,n,X,y):
    a = 0
    b = 0
    aval = []
    bval = []
    losses = []
    for i in range(n):
        aval.append(a)
        bval.append(b)
        predictions = a*X+b
        losses.append(lossFind(a,b,X,y))
        grad_a = np.mean(-2*X*(y - (a*X + b)))
        grad_b = np.mean(-2*(y - (a*X + b)))
        a-=grad_a*alpha
        b-=grad_b*alpha

    return(f"{a}x + {b}"), aval, bval, losses

func, aval, bval, losses = gradDesc(0.001,10000,X,noisy)

print(func)

# --- Build loss surface ---
A = np.linspace(min(aval), max(aval), 100)
B = np.linspace(min(bval), max(bval), 100)

A_grid, B_grid = np.meshgrid(A, B)
Z = np.zeros_like(A_grid)

for i in range(A_grid.shape[0]):
    for j in range(A_grid.shape[1]):
        Z[i, j] = lossFind(A_grid[i, j], B_grid[i, j], X, noisy)

# --- Plot ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(A_grid, B_grid, Z)

# plot gradient descent path
ax.plot(aval, bval, losses, linewidth=2)

ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('loss')

plt.show()
