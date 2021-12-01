"""
Move the hands and foots of atlas to target points 
"""
import pinocchio
import numpy as np
from utils.Visualizer import Visualizer
from os.path import abspath

if __name__ == "__main__":
    # Params
    model_dir = abspath('./urdf') 
    atlas_urdf = model_dir + '/atlas_description/urdf/atlas.urdf'
    T = 5.
    N = 100
    dt = T / N

    ## Give random joint angle series
    root = pinocchio.JointModelFreeFlyer()
    model = pinocchio.buildModelFromUrdf(atlas_urdf, root) # Generate atlas model from 
    data = model.createData()
    print(data)

    """
    q_traj = []
    q0 = pinocchio.normalize(model, np.zeros(model.nq))
    q0[2] = 1.0
    q_traj = [q0]
    for i in range(N):
        v = 0.05 * np.random.rand(model.nv)
        v = np.zeros(model.nv)
        q_traj.append(pinocchio.integrate(model, q_traj[-1], v)) 

    ## Visualize
    visualizer = Visualizer(atlas_urdf)
    visualizer.display_meshcat(dt=dt, q_traj=q_traj)
    """