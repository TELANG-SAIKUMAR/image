import os
import trimesh
from PIL import Image 

def optimize_and_validate_glb(input_path, output_path, max_bytes=2_000_000):
    # Load mesh or scene
    scene_or_mesh = trimesh.load(input_path, force='mesh')
    
    # Resize texture if present to keep file well under 2 MB
    if hasattr(scene_or_mesh.visual, 'material') and hasattr(scene_or_mesh.visual.material, 'baseColorTexture'):
        tex = scene_or_mesh.visual.material.baseColorTexture
        if tex is not None:
            tex = tex.convert("RGB")
            tex.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            scene_or_mesh.visual.material.baseColorTexture = tex

    # Center the model at the origin and normalize scale
    scene_or_mesh.vertices -= scene_or_mesh.centroid
    scale = 1.0 / max(scene_or_mesh.extents)
    scene_or_mesh.vertices *= scale

    # Export to binary glTF 2.0 (.glb)
    glb_data = scene_or_mesh.export(file_type='glb')
    with open(output_path, 'wb') as f:
        f.write(glb_data)

    size = os.path.getsize(output_path)
    print(f"Exported: {output_path} ({size:,} bytes)")
    assert size <= max_bytes, f"File exceeds 2 MB limit! ({size:,} > {max_bytes:,})"
    print("SUCCESS: Valid GLB under 2,000,000 bytes.")

if __name__ == "__main__":
    optimize_and_validate_glb("raw_iryna_bilyk.glb", "iryna_bilyk_q2662849.glb")