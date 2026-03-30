import numpy as np
import matplotlib.pyplot as plt

#initializing
X = np.linspace(1,10,10)
m = 3
c = 7
Y = m*X + c
mu = 2
noisy = Y + np.random.normal(mu,0.3,len(X))


#to find losses at each point for later plot and later terminator
def lossFind(m,c,X,y):
    loss = np.mean((y - (m*X+c))**2)
    return loss

#main func
def gradDesc(alpha,n,X,y):
    a = 0
    b = 0
    aval = []
    bval = []
    losses = []
    for i in range(n):
        aval.append(a)
        bval.append(b)
        losses.append(lossFind(a,b,X,y))
        grad_a = np.mean(-2*X*(y - (a*X + b)))
        grad_b = np.mean(-2*(y - (a*X + b)))
        a-=grad_a*alpha
        b-=grad_b*alpha
        
    return [(f"{a}x + {b}"), aval, bval, losses]

#creating set of output values for diff alphas
def alphaMc(startAlpha, variance, step, X, Y, itern):
    outputs = []
    end = startAlpha+variance
    while startAlpha<end:
        vals = gradDesc(startAlpha, itern, X, Y)
        outputs.append([startAlpha, vals[3]])
        startAlpha+=(step)

    return outputs

# plotting the alphas against the losses to find best alpha 
def outputInt(outputs):
 
    best = min(outputs, key = lambda pair: min(pair[1]))
    bestAlpha = best[0]
    losses = best[1]
    alphas = [l[0] for l in outputs]
    minLosses = [min(lis[1]) for lis in outputs ]
    plt.plot(alphas,minLosses)
    plt.ylabel("losses")
    plt.xlabel("alphas")
    plt.title("minimum losses at each alpha value to show how alpha impacts net loss")
    plt.show()
    print(str(best))

#func call 
outputInt(alphaMc(0.001, 0.01, 0.001, X, noisy, 1000))
