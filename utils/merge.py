#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import pickle
import numpy as np
import torch
import cv2
from pickle import UnpicklingError
import blosc

def loader(file):
    if str(file).endswith(".npy"):
        try:
            content = np.load(file, allow_pickle=True)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".dat"):
        try:
            with open(file, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".pkl"):
        try:
            with open(file, 'rb') as f:
                content = pickle.load(f)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    return None



def read_rgb_as_chw_float(rgb_path: Path, target_hw) -> np.ndarray:
    """BGR->RGB, resize to target_hw=(H,W), float32/255, CHW"""
    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {rgb_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = target_hw
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    ten = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    return ten.numpy()

def read_xyz_as_chw_float(npy_path: Path, target_hw) -> np.ndarray:
    """Read HWC xyz (meters) → resize each channel with nearest → CHW float32"""
    xyz = np.load(npy_path)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"Invalid XYZ shape: {xyz.shape}, file={npy_path}")
    H, W = target_hw
    if (xyz.shape[0], xyz.shape[1]) != (H, W):
        resized = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(3):
            resized[..., i] = cv2.resize(xyz[..., i], (W, H), interpolation=cv2.INTER_NEAREST)
        xyz = resized
    return xyz.transpose(2, 0, 1).astype(np.float32)

def read_transform(transform_path: Path) -> np.ndarray:
    """Read transform.txt → 4x4 float32"""
    M = np.loadtxt(transform_path, dtype=np.float64)
    M = np.atleast_2d(M)
    if M.shape == (3, 4):
        M = np.vstack([M, [0, 0, 0, 1]])
    if M.shape != (4, 4):
        raise ValueError(f"Unexpected transform shape {M.shape} at {transform_path}")
    return M.astype(np.float32)

def load_map_csv(map_csv: Path):
    """CSV: ep0001.pkl,ram_insertion_demo_1_20250820_193931  → dict[ep_name]=im_dir_name"""
    mapping = {}
    with open(map_csv, newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            ep, imdir = row[0].strip(), row[1].strip()
            mapping[ep] = imdir
    return mapping

def discover_pairs(orig_dir: Path, imag_dir: Path, map_csv: None):
    eps = sorted(orig_dir.glob("ep*.dat"))
    imds = sorted([p for p in imag_dir.iterdir() if p.is_dir()])

    if map_csv is not None:
        m = load_map_csv(map_csv)
        pairs = []
        im_lookup = {d.name: d for d in imds}
        for ep in eps:
            key = ep.name
            if key not in m:
                print(f"{key} has no mapping, skipped.")
                continue
            dname = m[key]
            if dname not in im_lookup:
                print(f"Mapped directory {dname} not found, skipping {key}")
                continue
            pairs.append((ep, im_lookup[dname]))
        return pairs

    n = min(len(eps), len(imds))
    if len(eps) != len(imds):
        print(f"Count mismatch: ep={len(eps)}, imagine_dirs={len(imds)}; pairing first {n}.")
    return list(zip(eps[:n], imds[:n]))

def append_three_fields(ep_in: Path, im_dir: Path, ep_out: Path):
    print(f"Processing: {ep_in.name}  +  {im_dir.name}  →  {ep_out.name}")
    # with open(ep_in, "rb") as f:
    #     ep = pickle.load(f)
    ep = loader(ep_in)

    if not (isinstance(ep, list) and len(ep) in (6, 9)):
        raise ValueError(f"Unexpected episode format in {ep_in}, len={len(ep)}")

    if len(ep) == 9:
        print(f"Already contains imagine fields, skipping: {ep_in.name}")
        return

    frame_ids, state_ls, action_list, attn_indices, gripper_list, trajectory_list = ep

    st = np.asarray(state_ls)
    H, W = st.shape[-2], st.shape[-1]
    target_hw = (H, W)

    img_path = im_dir / "imagine" / "fused_scene_rendered.png"
    pcd_path = im_dir / "imagine" / "fused_scene_filtered_dense_point_image_world.npy"
    T_path = im_dir / "imagine" / "transform.txt"

    if not img_path.exists():
        raise FileNotFoundError(f"Missing {img_path}")
    if not pcd_path.exists():
        raise FileNotFoundError(f"Missing {pcd_path}")
    if not T_path.exists():
        raise FileNotFoundError(f"Missing {T_path}")

    imagine_img = read_rgb_as_chw_float(img_path, target_hw)
    imagine_pcd = read_xyz_as_chw_float(pcd_path, target_hw)
    trans_matrix = read_transform(T_path)

    new_ep = [
        frame_ids,
        state_ls,
        action_list,
        attn_indices,
        gripper_list,
        trajectory_list,
        imagine_img,
        imagine_pcd,
        trans_matrix
    ]

    ep_out.parent.mkdir(parents=True, exist_ok=True)
    with open(ep_out, "wb") as f:
        f.write(blosc.compress(pickle.dumps(new_ep)))
    print(f"Saved episode: {ep_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name", help="Task name (e.g., phone_on_base)")
    parser.add_argument("split", help="Dataset split (train, val, test)")
    args = parser.parse_args()

    task = args.task_name
    split = args.split

    orig_dir = Path(f"./imagine2act/data/packaged/{split}/{task}+0")
    imag_dir = Path(f"./imagine2act/data/raw/{split}/{task}/variation0/episodes")
    out_dir = Path(f"./imagine2act/data/packaged/{split}_imagine/{task}+0")

    print(f"Input dir: {orig_dir}")
    print(f"Imagine dir: {imag_dir}")
    print(f"Output dir: {out_dir}")

    pairs = discover_pairs(orig_dir, imag_dir)
    if not pairs:
        print("No valid (episode, imagine_dir) pairs found.")
        return

    for ep_in, im_dir in pairs:
        ep_out = out_dir / ep_in.name
        append_three_fields(ep_in, im_dir, ep_out)

if __name__ == "__main__":
    main()
