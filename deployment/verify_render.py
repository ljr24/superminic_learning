"""Render a reference-motion smoke test. This is not a trained-policy result."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.rotations import mat_to_quat
os.makedirs('results', exist_ok=True)
m = MotionLib.from_dir('data/lafan1_g1', G1Kinematics('cpu'), 'cpu')
c = m.names.index('dance2_subject1')
start = int(m.clip_start[c])
n = min(150, int(m.clip_len[c]))
idx = torch.arange(start, start+n)
p, R, q = m.root_p[idx].numpy(), m.root_R[idx].numpy(), m.q[idx].numpy()
quat = mat_to_quat(m.root_R[idx]).numpy()
np.savez_compressed('results/verification_reference.npz', names=np.array(['dance2_subject1']),
 root_p=p[:,None], root_quat=quat[:,None], q=q[:,None], f=np.arange(n)[:,None], alive=np.ones((n,1),dtype=bool),
 ref_root_p=p, ref_root_R=R, ref_q=q, clip_start=np.array([0]), clip_len=np.array([n]))
print('Saved reference trajectory for rendering verification; not a trained policy.')
