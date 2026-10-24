import os
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter

IMG_PATH = 'cropped_face.png'
OUT_GLB = 'iryna_bilyk_q2662849.glb'

# 1. Load image and resize for clean mesh topology + crisp UV texture
img = Image.open(IMG_PATH).convert('RGB')
tex_img = img.resize((1024, 1024), Image.Resampling.LANCZOS)

# Grid resolution (180x180 = ~64,000 triangles, simplifies cleanly under 2MB)
N = 180
small_rgb = np.asarray(img.resize((N, N), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
lum = 0.299 * small_rgb[:, :, 0] + 0.587 * small_rgb[:, :, 1] + 0.114 * small_rgb[:, :, 2]

# 2. Build anatomical 3D cranial/facial dome + photometric depth relief
y_grid, x_grid = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing='ij')

# Ellipsoid base for human head/face volume
rx, ry = 0.78, 0.92
r2 = (x_grid / rx)**2 + (y_grid / ry)**2
head_dome = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0)) * 0.42

# Nose bridge, cheek, brow, and lip prominence
nose_ridge = np.exp(-((x_grid / 0.12)**2 + ((y_grid + 0.02) / 0.28)**2)) * 0.11
chin_cheek = np.exp(-((x_grid / 0.48)**2 + ((y_grid - 0.18) / 0.45)**2)) * 0.06
smooth_lum = gaussian_filter(lum, sigma=2.2)
fine_detail = (lum - gaussian_filter(lum, sigma=6.0)) * 0.035

z_front = head_dome + nose_ridge + chin_cheek + (smooth_lum * 0.08) + fine_detail
z_front = gaussian_filter(z_front, sigma=0.8)

# Mask out background corners smoothly
mask = np.clip(1.0 - (r2 ** 1.8), 0.0, 1.0)
z_front *= mask

# 3. Create vertices and UV coordinates
verts = np.stack([x_grid, -y_grid, z_front], axis=-1).reshape(-1, 3)
u_grid, v_grid = np.meshgrid(np.linspace(0, 1, N), np.linspace(1, 0, N), indexing='ij')
uvs = np.stack([u_grid, v_grid], axis=-1).reshape(-1, 2)

# Build triangle faces
faces = []
for i in range(N - 1):
    for j in range(N - 1):
        v0 = i * N + j
        v1 = v0 + 1
        v2 = (i + 1) * N + j
        v3 = v2 + 1
        faces.append([v0, v2, v1])
        faces.append([v1, v2, v3])
faces = np.array(faces, dtype=np.int64)

# 4. Create Textured Trimesh and export to GLB
material = trimesh.visual.material.PBRMaterial(
    baseColorTexture=tex_img,
    metallicFactor=0.0,
    roughnessFactor=0.85
)
visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=True)

# Center and normalize scale
mesh.vertices -= mesh.centroid
mesh.vertices *= (1.0 / max(mesh.extents))

glb_bytes = mesh.export(file_type='glb')
with open(OUT_GLB, 'wb') as f:
    f.write(glb_bytes)

size = os.path.getsize(OUT_GLB)
print(f'SUCCESS! Generated {OUT_GLB} ({size:,} bytes / {size/1024:.1f} KB)')
