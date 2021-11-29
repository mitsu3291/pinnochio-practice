import pinocchio
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
from utils.Visualizer import Visualizer
from os.path import abspath

if __name__ == "__main__":
    # Params
    model_dir = abspath('./urdf') 
    iiwa14_urdf = model_dir + '/iiwa_description/urdf/iiwa14.urdf'
    T = 5.
    N = 100
    dt = T / N

    ## Give random joint angle series
    model = pinocchio.buildModelFromUrdf(iiwa14_urdf) # Generate iiwa14 model (7DOF) from URDF

    q_traj = []
    q_traj.append(0.1 * pinocchio.randomConfiguration(model)) # Generate random joint angle
    for i in range(N):
        v = 0.05 * np.random.rand(7)
        q_traj.append(pinocchio.integrate(model, q_traj[-1], v)) 

    ## Visualize
    visualizer = Visualizer(iiwa14_urdf) # Create Visualizer object
    visualizer.display_meshcat(dt=dt, q_traj=q_traj)