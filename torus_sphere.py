# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:56:45 2024

@author: abel_
"""

import numpy as np
from scipy.spatial.distance import cdist

########SPHERE

def sample_points_on_sphere(num_points, r=1):
    points = []
    for _ in range(num_points):
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)

        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)

        x = r*np.sin(phi) * np.cos(theta)
        y = r*np.sin(phi) * np.sin(theta)
        z = r*np.cos(phi)

        points.append((x, y, z))

    return np.array(points)

def uniform_points_on_sphere(num_points_on_equator=10, r=1):
    points = []
    for i in range(num_points_on_equator+1):
        phi = i*np.pi/num_points_on_equator
        if np.sin(phi)==0:
            num_points_on_phi_ring=1
        else:
            num_points_on_phi_ring = abs(int(num_points_on_equator/r*np.sin(phi)))
        if i==num_points_on_equator:
            num_points_on_phi_ring=1
        for j in range(num_points_on_phi_ring):
            theta = j*2*np.pi/num_points_on_phi_ring
            x = r*np.sin(phi) * np.cos(theta)
            y = r*np.sin(phi) * np.sin(theta)
            z = r*np.cos(phi)
            points.append((x, y, z))
            
    return np.array(points)

def grid_on_sphere():
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Then, stack these flattened arrays together and transpose to get (N, 3)
    points = np.vstack((x_flat, y_flat, z_flat)).T
    return points





########TORUS

def torus_4d_to_3d_thetas(theta1, theta2, r1=1, r2=.25):
    # Convert angles to 3D Cartesian coordinates
    x = (r1+r2*np.cos(theta1))*np.cos(theta2);
    y = (r1+r2*np.cos(theta1))*np.sin(theta2);
    z = r2*np.sin(theta1);

    return x,y,z


def torus_4d_to_thetas(points, r1=1, r2=.25):
    # Convert 4D Cartesian coordinates to angles
    theta1 = np.arctan2(points[:,1],points[:,0])
    theta2 = np.arctan2(points[:,3],points[:,2])
    return theta1,theta2


def torus_4d_to_3d(points, r1=1, r2=.25):
    # Convert 4D Cartesian coordinates to angles
    theta1 = np.arctan2(points[:,1],points[:,0])
    theta2 = np.arctan2(points[:,3],points[:,2])
    
    # Convert angles to 3D Cartesian coordinates
    x = (r1+r2*np.cos(theta1))*np.cos(theta2);
    y = (r1+r2*np.cos(theta1))*np.sin(theta2);
    z = r2*np.sin(theta1);

    return x,y,z


def get_manifold_from_closest_projections_torus(trajectories_flat, wo, npoints=128):

    n_rec = wo.shape[0]
    xs = np.arange(-np.pi, np.pi, 2*np.pi/npoints)
    xs = np.append(xs, -np.pi)
    ys = np.dot(trajectories_flat.reshape((-1,n_rec)), wo)
    circle_points = np.array([np.cos(xs), np.sin(xs)]).T
    torus_points = np.array(np.meshgrid(circle_points,circle_points))
    dists = cdist(torus_points.reshape((-1,4)), ys)
    csx2 = []
    for i in range(xs.shape[0]):
        csx2.append(trajectories_flat[np.argmin(dists[i,:]),:])
    csx2 = np.array(csx2)
    csx2_proj2 = np.dot(csx2, wo)
    return csx2, csx2_proj2




def grid_on_3dtorus(R=1.0, r=0.5, u_res=20j, v_res=10j):
    # Create a meshgrid for u and v angles
    u, v = np.mgrid[0:2*np.pi:u_res, 0:2*np.pi:v_res]

    # Parametric equations for the torus
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    # Flatten the arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Stack and transpose to get (N, 3) points array
    points = np.vstack((x_flat, y_flat, z_flat)).T
    
    return points


def grid_on_4dtorus(R=1.0, r=0.5, u_res=20j, v_res=20j):
    # Create a meshgrid for u1, u2, v1, v2 angles
    u, v = np.mgrid[0:2*np.pi:u_res, 0:2*np.pi:v_res]

    # Parametric equations for the 4D torus
    w1 = R * np.cos(v)
    w2 = R * np.sin(v)
    w3 = r * np.cos(u) 
    w4 = r * np.sin(u) 

    # Flatten the arrays
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()
    w3_flat = w3.flatten()
    w4_flat = w4.flatten()

    # Stack and transpose to get (N, 4) points array
    points = np.vstack((w1_flat, w2_flat, w3_flat, w4_flat)).T
    
    return points


#simulate trajectories on torus
# points = grid_on_4dtorus(u_res=100j, v_res=100j)
# points_out = points*1.2
# points_init_nd = np.dot(D, points_out.T).T; Nsims=points_out.shape[0];
# trajs = simulate_network_ntimes(Nsims, W, 0, nonlinearity_ode=tanh_ode, tau=10, maxT=250, tsteps=5001,
#   mlrnn=False, y0s=points_init_nd);
# trajs_proj_outside = np.dot(trajs,D); 
# points_in = points*0.8
# points_init_nd = np.dot(D, points_in.T).T; 
# trajs = simulate_network_ntimes(Nsims, W, 0, nonlinearity_ode=tanh_ode, tau=10, maxT=250, tsteps=5001,
#   mlrnn=False, y0s=points_init_nd);
# trajs_proj_inside = np.dot(trajs,D);

#plot torus and fps
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d', computed_zorder=True);
# x, y, z = torus_4d_to_3d(xstars_2, r1=1, r2=.3)
# ax.scatter(x, y, z, color=stab_colors[stabilist])

# x, y, z = torus_4d_to_3d(trajs_proj_outside[:,-1,:], r1=1, r2=.3)
# x = x.reshape((20,20))
# y = y.reshape((20,20))
# z = z.reshape((20,20))
# ax.plot_wireframe(x, y, z, color="k")
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax.zaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_zlim([-1,1]);  plt.savefig(folder+'/torus_fps.pdf')


#plot torus wire
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d');
# x, y, z = torus_4d_to_3d(trajs_proj_outside[:,0,:], r1=1, r2=.4)
# x = x.reshape((20,20))
# y = y.reshape((20,20))
# z = z.reshape((20,20))
# ax.plot_wireframe(x, y, z, color="b")

# x, y, z = torus_4d_to_3d(trajs_proj_outside[:,-1,:], r1=1, r2=.3)
# x = x.reshape((20,20))
# y = y.reshape((20,20))
# z = z.reshape((20,20))
# ax.plot_wireframe(x, y, z, color="r")
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax.zaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_zlim([-1,1]); ax.plot_wireframe(x, y, z, color="r"); plt.savefig(folder+'/spheretorus_start_end_big.pdf')