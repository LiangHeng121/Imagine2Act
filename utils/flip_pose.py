import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("pose_txt", help="Input: original pose.txt (with positive fx/fy)")
parser.add_argument("output_pose_txt", help="Output: flipped pose.txt (for negative fx/fy)")
args = parser.parse_args()

T = np.loadtxt(args.pose_txt)

S = np.diag([-1, -1, 1, 1])  # flip x and y axes

T_new = S @ T

np.savetxt(args.output_pose_txt, T_new, fmt="%.8f")
print(f"Pose file saved to: {args.output_pose_txt}")