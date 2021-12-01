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
    model = pinocchio.buildModelFromUrdf(atlas_urdf, root) # Generate atlas model from URDF
    """
    print(model.getFrameId('r_arm_wry2'))
    print(model.getFrameId('l_arm_wry2'))
    print(model.getFrameId('r_hand'))
    print(model.getFrameId('l_hand'))
    print(model.getFrameId('r_ufarm'))
    print(model.getFrameId('l_uarm'))
    print(model.getFrameId('r_talus'))
    print(model.getFrameId('l_talus'))
    """

    #for i in range(len(model.joints)):
        #print(f"Number : {i}")
        #print(model.names[i])
        #print(model.joints[i])
    for i in range(len(model.frames)):
        print(f"Number : {i}")
        print(model.frames[i].name)

    print(model.getFrameId('universe'))
    print(model.getFrameId('l_clav'))

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