# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:53:01 2025

@author: guillermo_
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import ortho_group


def EMPJ(sigma, dsigma, fixedp, Utarg, Starg, tau):
  k = fixedp.shape[1]
  N = fixedp.shape[0]
  Af0 = np.row_stack([sigma(fixedp[:,i]) for i in range(k)])
  Af1 = np.row_stack([Utarg[:,:,i].T @ np.diag(dsigma(fixedp[:,i])) for i in range(k)])
  A = np.row_stack([Af0, Af1])
  Bf0 = np.row_stack([fixedp[:,i] for i in range(k)])
  Bf1 = np.row_stack([tau * (Starg[:,:,i] + 1/tau*np.eye(Starg.shape[0])) @ Utarg[:,:,i].T for i in range(k)])
  B = np.row_stack([Bf0, Bf1])
  W = np.linalg.lstsq(A, B, rcond=-1)[0]
  print(np.sum(A @ W - B))
  return W


def euler(F, iv, tsteps, dt):
  X = np.zeros((iv.shape[0], tsteps))
  X[:,0] = iv
  for t in np.arange(tsteps-1)+1:
    dX = F(X[:,t-1])
    X[:,t] = X[:,t-1] + dt*dX
  return X.T


cat = 'tanh'
if cat == 'tanh':
  sigma = lambda x: np.tanh(x)
  dsigma = lambda x: 1 - np.tanh(x)**2
else:
  sigma = lambda x: 1/(1+np.exp(-x))
  dsigma = lambda x: sigma(x)*(1 - sigma(x))
  
  
### FLAT RING ATTRACTOR ###
cat = 'torus' # 'purecode'
if cat == 'flatring':
  N = 10
  k = 3
  tau = 0.05

  # Encoder matrix
  np.random.seed(0)
  D = ortho_group.rvs(N)[:,:2]
  # with open('D.json', 'w') as json_file:
  #   json.dump(D.tolist(), json_file)

  th = np.linspace(0.1, 2*np.pi, k, endpoint=False)
  x = np.cos(th)
  y = np.sin(th)
  ringp = np.array([x, y])
  ringv = np.array([[-y,x], [x,y]])

  fixedp = D @ ringp
  Utarg = np.dot(D, ringv)
  Utarg = Utarg/np.linalg.norm(Utarg,axis=0)
  Starg = np.tile(np.diag([+1, -1/tau])[:,:,np.newaxis],k)
  print(Starg)
  print(Starg.shape)

elif cat == 'purecode':
  N = 3
  tau = 0.05
  maxa = 0.8

  th = np.linspace(0.1, 2*np.pi, N, endpoint=False)
  x = np.cos(th)
  y = np.sin(th)
  ringp = np.array([x, y])
  ringv = np.array([[-y,x], [x,y]])

  D = ringp.T
  D = D/np.linalg.norm(D,axis=0)

  fixedp = maxa*np.eye(N)
  Utarg = np.dot(D, ringv)
  Utarg = Utarg/np.linalg.norm(Utarg,axis=0)
  Starg = np.tile(np.diag([-1/tau, -1/tau])[:,:,np.newaxis],N)
elif cat == 'sparse':
  N = 3
  tau = 0.05
  maxa = 0.8

  sqrt2 = np.sqrt(2)
  D = np.array([[-1/sqrt2, 1/sqrt2, 0],[-1/(2*sqrt2), -1/(2*sqrt2), 1/sqrt2]]).T

  fixedp = maxa*np.eye(N)
  Utarg = np.array([np.delete(np.eye(N), i, axis=1).T for i in range(N)]).T # (N, v, k)
  print(Utarg.shape)
  print(Utarg[:,:,0])
  Starg = np.tile(np.diag([-1, -1])[:,:,np.newaxis],N)

elif cat == 'otherring' :
  N = 3
  tau = 0.05
  maxa = 0.2

  sqrt2 = np.sqrt(2)
  D = np.array([[-1/sqrt2, 1/sqrt2, 0],[-1/(2*sqrt2), -1/(2*sqrt2), 1/sqrt2]]).T

  fixedp = maxa*np.eye(N)
  Utarg = np.array([((np.delete(np.eye(N), i, axis=1) - np.eye(N)[i,:,np.newaxis])/np.sqrt(2)).T for i in range(N)]).T # (N, v, k)
  print(Utarg.shape)
  print(Utarg[:,:,0])
  print(Utarg[:,:,1])
  print(Utarg[:,:,2])
  Starg = np.tile(np.diag([1, -1])[:,:,np.newaxis],N)
  Starg[:,:,1] = -Starg[:,:,1]
  print(Starg)
  print(Starg.shape)

elif cat == 'torus':
  N = 80
  tau = 0.05
  sqrtk = 5
  l = 0

  np.random.seed(2)
  D = ortho_group.rvs(N)[:,:4]

  th1 = np.linspace(0, 2*np.pi, sqrtk, endpoint=False)
  th2 = np.linspace(0, 2*np.pi, sqrtk , endpoint = False)
  th1, th2 = np.meshgrid(th1, th2)
  th1 = th1.flatten()
  th2 = th2.flatten()

  x = np.cos(th1)
  y = np.sin(th1)
  zx = np.cos(th2)
  zy = np.sin(th2)
  ringp = np.vstack([x, y, zx, zy])

  ringp = ringp / np.linalg.norm(ringp, axis=0)

  zvec = np.zeros_like(x)
  ringv = np.array([[-y, x, zvec, zvec], [zvec, zvec, -zy, zx], [x, y, zvec, zvec], [zvec, zvec, zx, zy]])

  print(ringp.shape)
  print(D.shape)
  fixedp = D @ ringp
  Utarg = np.dot(D, ringv)
  Utarg = Utarg/np.linalg.norm(Utarg,axis=0)
  Starg = np.tile(np.diag([l, l, -1/tau, -1/tau])[:,:,np.newaxis],sqrtk**2)
  print(Starg)
  print(Starg.shape)

elif cat == 'sphere':
  N = 100
  tau = 0.05
  angles = 10
  r = 2
  k = 2*r*angles
  l = 0

  np.random.seed(0)
  D = ortho_group.rvs(N)[:,:3]



  th = np.linspace(0.1, 2*np.pi, angles, endpoint=False)
  rho = np.linspace(0.2, 0.9, r, endpoint=True)

  x = np.tile((rho[:,np.newaxis] @ np.cos(th)[np.newaxis,:]).flatten(), 2)
  y = np.tile((rho[:,np.newaxis] @ np.sin(th)[np.newaxis,:]).flatten(), 2)
  zp = np.sqrt(1 - (x[:r*angles]**2 +  y[:r*angles]**2))
  z = np.hstack([zp, -zp])


  ringp = np.vstack([x, y, z])
  ringp = ringp / np.linalg.norm(ringp, axis=0)
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title('3D Vis')
  ax.scatter(ringp[0,:],ringp[1,:],ringp[2,:])
  ax.set_xlim([-1,1])
  ax.set_ylim([1,-1])
  ax.set_zlim([-1,1])

  zvec = np.zeros_like(x)
  ringv = np.array([[-y, x, zvec], [-z, zvec, x], ringp])
  print(ringv.shape)
  #ringv = ringv / np.linalg.norm(ringv, axis=1)

  print(ringp)
  print(D.shape)
  fixedp = D @ ringp
  Utarg = np.dot(D, ringv)
  Utarg = Utarg/np.linalg.norm(Utarg,axis=0)
  Starg = np.tile(np.diag([l, l, -1/tau])[:,:,np.newaxis],k)
  print(Starg)
  print(Starg.shape)

else:
  N = 100
  tau = 0.05
  k = 40
  l = 0

  np.random.seed(3)
  D = ortho_group.rvs(N)[:,:3]

  th = np.random.uniform(0, 2*np.pi, k)
  rho = np.random.uniform(0, 1, k)

  x = np.tile(rho * np.cos(th), 2)
  y = np.tile(rho * np.sin(th), 2)
  zp = np.sqrt(1 - (x[:k]**2 +  y[:k]**2))
  z = np.hstack([zp, -zp])


  ringp = np.vstack([x, y, z])
  ringp = ringp / np.linalg.norm(ringp, axis=0)
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title('3D Vis')
  ax.scatter(ringp[0,:],ringp[1,:],ringp[2,:])
  ax.set_xlim([-1,1])
  ax.set_ylim([1,-1])
  ax.set_zlim([-1,1])

  zvec = np.zeros_like(x)
  ringv = np.array([[-y, x, zvec], [-z, zvec, x], ringp])

  print(ringp)
  print(D.shape)
  fixedp = D @ ringp
  Utarg = np.dot(D, ringv)
  Utarg = Utarg/np.linalg.norm(Utarg,axis=0)
  Starg = np.tile(np.diag([l, l, -1/tau])[:,:,np.newaxis],2*k)
  print(Starg)
  print(Starg.shape)


W = EMPJ(sigma, dsigma, fixedp, Utarg, Starg, tau)
with open('W_torus.json', 'w') as json_file:
  json.dump(W.tolist(), json_file)
with open('D_torus.json', 'w') as json_file:
  json.dump(D.tolist(), json_file)
# print(D)
# print(W)


F = lambda X: 1/tau * (-X + W.T @ sigma(X))

i_vec = np.linspace(-1,1,5)
X, Y = np.meshgrid(i_vec,i_vec+0.05)

if cat == 'torus':
  x_ = np.linspace(-1, 1, 5)
  y_ = np.linspace(-1, 1, 5)
  z_ = np.linspace(-1, 1, 5)

  x, y, z, w = np.meshgrid(x_, y_, z_, z_, indexing='ij')
  pts = np.vstack([x.flatten(), y.flatten(), z.flatten(), w.flatten()])
  iv = D @ pts
elif cat == 'sphere' or cat == 'randsphere':
  angle = np.random.uniform(0, 2*np.pi, 10)
  rho = np.random.uniform(0, 1, 10)
  x = np.tile(rho*np.cos(angle), 2)
  y = np.tile(rho*np.sin(angle), 2)
  zp = np.sqrt(1 - (x[:10]**2 + y[:10]**2))
  z = np.hstack([zp, -zp])

  x = np.random.uniform(-1, 1, 20)
  y = np.random.uniform(-1, 1, 20)
  z = np.random.uniform(-1, 1, 20)

  x_ = np.linspace(-1, 1, 8)
  y_ = np.linspace(-1, 1, 8)
  z_ = np.linspace(-1, 1, 8)

  x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
  pts = np.vstack([x.flatten(), y.flatten(), z.flatten()])
  iv = D @ pts
else:
  iv = D @ np.stack((X.flatten(), Y.flatten()), axis=0)
  if cat != 'flatring' and cat != 'torus': iv += np.ones_like(iv)*maxa/N

trajs = iv.shape[1]
dt = 1e-3
T = 20
tsteps = int(T/dt)



traj = np.zeros((tsteps, N, trajs))
for i in np.arange(trajs):
  traj[:,:,i] = euler(F, iv[:,i], tsteps, dt)

traj2d = D.T @ traj


if N >= 3:
  fixedp3d = D.T @ fixedp
  iv3d = D.T @ iv
  #print(np.linalg.norm(iv, axis=0))
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title('3D Vis')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_xlim(-1, 1)
  ax.set_ylim(1, -1)
  ax.set_zlim(-1, 1)
  #ax.quiver(fixedp3d[0,:], fixedp3d[1,:], fixedp3d[2,:], Utarg[0,:,:], Utarg[1,:,:], Utarg[2,:,:], length=0.8, zorder=100, color='b')
  ax.plot(np.linspace(-1,1,20), np.zeros(20), np.zeros(20), color='k')
  ax.plot(np.zeros(20), np.linspace(-1,1,20), np.zeros(20), color='k')
  ax.plot(np.zeros(20), np.zeros(20), np.linspace(-1,1,20), color='k')
  for i in np.arange(trajs):
    ax.plot(traj2d[:,0,i], traj2d[:,1,i], traj2d[:,2,i], color='r', alpha=0.5)
    ax.scatter(iv3d[0,i], iv3d[1,i], iv3d[2,i], color='r', alpha=0.5, zorder=20)
    ax.scatter(traj2d[-1,0,i], traj2d[-1,1,i], traj2d[-1,2,i], color='b', alpha=0.5)
  #ax.scatter(fixedp3d[0,:], fixedp3d[1,:], fixedp3d[2,:], zorder=100, color='g')
  plt.grid()
  plt.show()
  fig.savefig("torus3d.pdf", bbox_inches='tight')
  print("dx(x_i) = " + str(np.linalg.norm(W.T @ sigma(fixedp) - fixedp, axis=0)))
  for i in np.arange(fixedp.shape[1]):
    J = 1/tau* (-np.eye(N) + W.T @ np.diag(dsigma(fixedp[:,i])))
    lamb, _ = np.linalg.eig(J)
    print("Eigen values J(x_" + str(i) + ") =")
    for l in lamb:
      print("{:.2f} + {:.2f}i".format(l.real, l.imag))