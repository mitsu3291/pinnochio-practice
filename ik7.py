#
# 7自由度マニピュレータの逆運動学
#

from __future__ import print_function

import numpy as np
from sys import argv
from os.path import abspath

import pinocchio

model_dir = abspath('./urdf') 
iiwa14_urdf = model_dir + '/iiwa_description/urdf/iiwa14.urdf'
model = pinocchio.buildModelFromUrdf(iiwa14_urdf) # URDF から iiwa14 モデル (7DOF) を生成
data  = model.createData() # pinocchio では上で定義した model オブジェクトと data オブジェクトを計算に使用

frame_id = 22 # エンドエフェクターフレームの id
q_ref = 0.1 * pinocchio.randomConfiguration(model) # ランダムな関節角を生成
pinocchio.forwardKinematics(model, data, q_ref) # q_ref の値に基づいて順動力学を計算
pinocchio.updateFramePlacements(model, data) # 順動力学計算に基づいてフレームの位置・回転行列を更新
x_ref = data.oMf[frame_id].copy() # エンドエフェクターフレームの位置・回転行列を取得

q        = np.zeros(model.nq) # 初期関節角度は 0 に設定
eps      = 1e-4 # 誤差が eps 以下で収束
iter_max = 1000 # Levenberg-Marquardt 法の最大反復回数
alpha    = 1.0 # Levenberg-Marquardt 法のステップサイズ
damp0    = 1e-3 # Levenberg-Marquardt 法の正則化項

i=0
while True:
    pinocchio.forwardKinematics(model, data, q) # q の値に基づいて順動力学を計算
    pinocchio.updateFramePlacements(model, data) # 順動力学計算に基づいてフレームの位置・回転行列を更新
    dMi = x_ref.actInv(data.oMf[frame_id])  # 目標の SE3 と 現在の SE3 の差を表す SE3 を計算．差は逆行列の積によって表される．
    err = pinocchio.log(dMi).vector # 差を表す SE3 を6次元ベクトルに変換
    if np.linalg.norm(err) < eps: # 誤差が小さければ反復終了
        success = True
        break
    if i >= iter_max:
        success = False
        break
    J = pinocchio.computeFrameJacobian(model, data, q, frame_id, pinocchio.LOCAL_WORLD_ALIGNED) # frame_id の SE3 に対応するヤコビ行列を取得．LOCAL_WORLD_ALIGNED はワールド座標系でのヤコビ行列を意味する．
    damp = damp0 + np.linalg.norm(err)
    dq = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err)) # Levenberg-Marquardt 法で q の更新ステップを決定．
    q = pinocchio.integrate(model, q, dq*alpha) # q を dq と alpha（ステップサイズ）に基づいて更新．
    if not i % 10:
        print('%d: error = %s' % (i, err.T))
    i += 1

if success:
    print("Convergence achieved!")
else:
    print("Warning: the iterative algorithm has not reached convergence to the desired precision")

print('\nq_ref: %s' % q_ref.flatten().tolist())
print('x_ref.trans: %s' % x_ref.translation.flatten().tolist())
print('x_ref.R: %s' % x_ref.rotation.flatten().tolist())

print('\nq: %s' % q.flatten().tolist())
print('x.trans: %s' % data.oMf[frame_id].translation.flatten().tolist())
print('x.R: %s' % data.oMf[frame_id].rotation.flatten().tolist())

print('\nfinal error: %s' % err.T)