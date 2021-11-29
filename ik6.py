"""
Solve inverse kinematics of 6 DOF Manipulator numerically
"""

import numpy as np
import pinocchio
import inspect

model = pinocchio.buildSampleModelManipulator() # Create robot arm model
data = model.createData()

frame_id = 13 # Id of the end-effector
q_ref = 0.1*pinocchio.randomConfiguration(model)
pinocchio.forwardKinematics(model, data, q_ref) # Calc forward kinematics
pinocchio.updateFramePlacements(model, data) # Update the position of the frame and rotation matrix
x_ref = data.oMf[frame_id].copy() # Get the position of frame and rotation matrix

q        = np.zeros(model.nq) # Initialize joint angles by 0
eps      = 1e-4 # The threshold judges conversion
iter_max = 1000 # The number of max iteration of Levenberg-Marquardt method
alpha    = 1.0  # The step size of Levenberg-Marquardt method
damp0    = 1e-3 # The Regularization term of Levenberg-Marquardt method

i=0
while True:
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.updateFramePlacements(model, data)
    dMi = x_ref.actInv(data.oMf[frame_id])
    err = pinocchio.log(dMi).vector
    if np.linalg.norm(err) < eps:
        Is_success = True
        break
    if i >= iter_max:
        Is_success = False
        break
    J = pinocchio.computeFrameJacobian(model, data, q, frame_id, pinocchio.LOCAL_WORLD_ALIGNED)
    damp = damp0 + np.linalg.norm(err)
    dq = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp*np.eye(6),err))
    q = pinocchio.integrate(model, q, dq*alpha)
    if not i % 10:
        print('%d: error = %s' % (i, err.T))
    i += 1

if Is_success:
    print("Convergence achieved!")
else:
    print("Warinig: the iterative algorithm has not reached convergence to the desired precision")

print('\nq_ref: %s' % q_ref.flatten().tolist())
print('x_ref.trans: %s' % x_ref.translation.flatten().tolist())
print('x_ref.R: %s' % x_ref.rotation.flatten().tolist())

print('\nq: %s' % q.flatten().tolist())
print('x.trans: %s' % data.oMf[frame_id].translation.flatten().tolist())
print('x.R: %s' % data.oMf[frame_id].rotation.flatten().tolist())

print('\nfinal error: %s' % err.T)

print(type(model))
for x in inspect.getmembers(model, inspect.ismethod):
    print(x[0])

for i in range(len(model.frames)):
    print(f"Number : {i}")
    print(model.frames[i])

print(model.getFrameId("lowerarm_body"))