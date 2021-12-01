"""
Move the hands and foots of atlas to target points 
"""
from numpy import linalg
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

    ## position reference
    # x_ref : [middle_torso_ref, l_hand_ref, r_hand_ref, l_foot_ref, r_foot_ref]
    offset = np.array([0.0, 0, 0.5]*4) # [depth, width, height]
    x_ref = np.array([-0.5, 0, 0, 0, 0.5, 0.6, 0, -0.5, 0.6, 0, 0.5, -0.5, 0, -0.5, -0.5])
    x_ref = np.array([0, 0.5, 0.9, 0, -0.5, 0.9, 0, 0.5, -0.5, 0, -0.5, -0.5])
    x_ref += offset

    q        = pinocchio.normalize(model, np.zeros(model.nq)) # 初期関節角度は 0 に設定
    eps      = 1e-4 # 誤差が eps 以下で収束
    iter_max = 1000 # Levenberg-Marquardt 法の最大反復回数
    alpha    = 1.0  # Levenberg-Marquardt 法のステップサイズ
    damp0    = 1e-3 # Levenberg-Marquardt 法の正則化項

    i=0
    while True:
        pinocchio.forwardKinematics(model, data, q) # q の値に基づいて順動力学を計算
        pinocchio.updateFramePlacements(model, data)  # 順動力学計算に基づいてフレームの位置・回転行列を更新
        x_now = np.array([])
        #x_now = np.concatenate([x_now, data.oMf[10].translation]) # add current middle torso position
        x_now = np.concatenate([x_now, data.oMf[21].translation]) # add current left hand position
        x_now = np.concatenate([x_now, data.oMf[63].translation]) # add current right hand position
        x_now = np.concatenate([x_now, data.oMf[80].translation]) # add current left foot position
        x_now = np.concatenate([x_now, data.oMf[92].translation]) # add current right foot position
        err = x_now - x_ref
        if np.linalg.norm(err) < eps: # 誤差が小さければ反復終了
            Is_success = True
            break
        if i >= iter_max:
            Is_success = False
            break
        #J = pinocchio.computeFrameJacobian(model, data, q, 10, pinocchio.LOCAL_WORLD_ALIGNED)[0:3, :]                      # add jacobian of middle torso position
        J = pinocchio.computeFrameJacobian(model, data, q, 21, pinocchio.LOCAL_WORLD_ALIGNED)[0:3, :]  
        #J = np.concatenate([J, pinocchio.computeFrameJacobian(model, data, q, 21, pinocchio.LOCAL_WORLD_ALIGNED)[0:3, :]]) # add jacobian of left hand position
        J = np.concatenate([J, pinocchio.computeFrameJacobian(model, data, q, 63, pinocchio.LOCAL_WORLD_ALIGNED)[0:3, :]]) # add jacobian of right hand position
        J = np.concatenate([J, pinocchio.computeFrameJacobian(model, data, q, 80, pinocchio.LOCAL_WORLD_ALIGNED)[0:3, :]]) # add jacobian of left foot position
        J = np.concatenate([J, pinocchio.computeFrameJacobian(model, data, q, 92, pinocchio.LOCAL_WORLD_ALIGNED)[0:3, :]]) # add jacobian of right foot position
        damp = damp0 + np.linalg.norm(err)
        dq = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(12), err)) # Levenberg-Marquardt 法で q の更新ステップを決定．
        q = pinocchio.integrate(model, q, dq*alpha) # q を dq と alpha（ステップサイズ）に基づいて更新．
        if not i % 10:
            print('%d: error = %s' % (i, err.T))
        i += 1

    if Is_success:
        print("Convergence achieved!")
    else:
        print("Warning: the iterative algorithm has not reached convergence to the desired precision")

    print('x_ref: %s' % x_ref.flatten().tolist())

    print('\nq: %s' % q.flatten().tolist())
    print('x_middle_torso : %s' % data.oMf[10].translation.flatten().tolist())
    print('x_left_hand : %s' % data.oMf[21].translation.flatten().tolist())
    print('x_right_hand : %s' % data.oMf[63].translation.flatten().tolist())
    print('x_left_foot : %s' % data.oMf[80].translation.flatten().tolist())
    print('x_right_foot : %s' % data.oMf[92].translation.flatten().tolist())

    print('\nfinal error: %s' % err.T)

    q_traj = []
    q0 = pinocchio.normalize(model, np.zeros(model.nq))
    q0[2] = 1.0
    q_traj = [q]
    for i in range(N):
        v = 0.05 * np.random.rand(model.nv)
        v = np.zeros(model.nv)
        q_traj.append(pinocchio.integrate(model, q_traj[-1], v)) 

    ## Visualize
    visualizer = Visualizer(atlas_urdf)
    visualizer.display_meshcat(dt=dt, q_traj=q_traj)