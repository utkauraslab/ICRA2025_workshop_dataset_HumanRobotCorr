## 🛠 How to Modify

- Check the required packages in `requirements.txt`
- Modify the following lines in `pcl_voxelization.py`:
  - **Line 189**: Set your `label_cvs_path`
  - **Line 190**: Set your `image_root_folder`

---

## 🚀 How to Run

```
python3 pcl_voxelization.py
```

- Running this script will generate a variable called `voxel_grid` (around **line 267**) which can be used for subsequent steps.
- To visualize the generated voxel grid, use:

```
o3d.visualization.draw_geometries([pcd, voxel_pcd])
```

---

## 🌊 How to Generate New Depth Images

This code works best with **high-quality depth images**. The original ones may not yield good voxelizations, so we recommend using **Depth Anything V2** to generate improved depth maps.

### Steps:

1. **Clone Depth Anything V2**:

```
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

2. **Download their pretrained model**  
   - Use the [small model](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#pre-trained-models) — it already gives good results.

3. Place everything inside your `Generate_Voxels/` directory.

4. Run the following to generate new depth images:

```
python3 get_depth_images.py
```