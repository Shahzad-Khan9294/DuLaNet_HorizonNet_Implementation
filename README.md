# Make Env using python pakage : python 3.7  -- EnvName [pano_ann_env]

# Core Python packages
pip install numpy==1.19.5 \
            scipy==1.3.0 \
            scikit-learn==0.24.2 \
            pillow==8.4.0 \
            scikit-image==0.18.3 \
            tqdm==4.64.1 \
            tensorboardx==2.0 \
            opencv-python==4.2.0.32 \
            pylsd-nova==1.2.0 \
            open3d==0.13.0 \
            shapely==1.7.1 \
            matplotlib==3.3.4 \
            networkx==2.5 \
            imageio==2.9.0
            
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install GitPython


#HorizonNet - Set 360 degree room images in right orientation (all walls parallel/perpendicular)
#DuLaNet - 
1. Layout Prediction [Detects important elements like corners, boundaries, and wall edges]
2. 3D Layout ConstructionUses [geometric reasoning to reconstruct the room in 3D, generating walls, planes, and room height]
3. Dual Projection [Ceiling-view projection (top-down view of the room) & Panorama-view projection (the original equirectangular image)]


HERE ARE THE RESULTS!

![DulaNet_Result](https://github.com/user-attachments/assets/c999c34a-228f-47ec-a372-884a3edf491f)
