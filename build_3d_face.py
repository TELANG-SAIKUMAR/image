import os
import shutil
import trimesh
from PIL import Image
from gradio_client import Client, handle_file

MAX_BYTES = 2_000_000
INPUT_IMAGE = "input.png"
CROPPED_IMAGE = "cropped_face.png"
RAW_GLB = "raw_model.glb"
FINAL_GLB = "iryna_bilyk_q2662849.glb"

def step1_crop_portrait(input_path, output_path):
    print("1. Cropping portrait around head and hair...")
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    # Crop tightly around the head, hair, and neck (top ~65% of image)
    left = int(w * 0.12)
    top = int(h * 0.02)
    right = int(w * 0.88)
    bottom = int(h * 0.65)
    cropped = img.crop((left, top, right, bottom))
    cropped.save(output_path)
    print(f"   Saved cropped image to {output_path}")

def step2_generate_raw_glb(cropped_path, raw_glb_path):
    print("2. Generating 3D mesh via Hugging Face API (TripoSR / Trellis)...")
    try:
        # Connects to a public Image-to-3D space on Hugging Face
        client = Client("stabilityai/TripoSR")
        result = client.predict(
            handle_file(cropped_path),
            True,   # Remove background automatically
            0.85,   # Foreground ratio
            256,    # Marching cubes resolution
            api_name="/run"
        )
        # Copy the returned mesh file to our working folder
        generated_file = result[1] if isinstance(result, (list, tuple)) else result
        shutil.copy(generated_file, raw_glb_path)
        print(f"   Raw 3D model saved to {raw_glb_path}")
    except Exception as e:
        print(f"   API Space busy or offline ({e}).")
        print("   -> Please upload 'cropped_face.png' manually to https://huggingface.co/spaces/trellis-community/TRELLIS")
        print("   -> Download the .glb file into this folder as 'raw_model.glb' and re-run this script.")
        exit(1)

def step3_optimize_and_validate(raw_glb_path, final_glb_path):
    print("3. Optimizing mesh and texture to stay under 2,000,000 bytes...")
    mesh = trimesh.load(raw_glb_path, force='mesh')

    # Center at origin and normalize size
    mesh.vertices -= mesh.centroid
    scale = 1.0 / max(mesh.extents)
    mesh.vertices *= scale

    # Simplify geometry if face count is too high (> 25,000 triangles)
    if len(mesh.faces) > 25000:
        print(f"   Simplifying mesh from {len(mesh.faces)} faces to 25,000 faces...")
        mesh = mesh.simplify_quadric_decimation(25000)

    # Compress texture resolution if present
    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'baseColorTexture'):
        tex = mesh.visual.material.baseColorTexture
        if tex is not None:
            tex = tex.convert("RGB")
            tex.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            mesh.visual.material.baseColorTexture = tex

    # Export to binary glTF 2.0 (.glb)
    glb_bytes = mesh.export(file_type='glb')
    with open(final_glb_path, 'wb') as f:
        f.write(glb_bytes)

    size = os.path.getsize(final_glb_path)
    print(f"   Exported: {final_glb_path} ({size:,} bytes)")

    if size > MAX_BYTES:
        raise ValueError(f"File is still over 2 MB ({size:,} bytes). Run the npx gltf-transform command below.")
    print("SUCCESS! Your GLB is valid and under the 2 MB limit.")

if __name__ == "__main__":
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"Please save the photo as '{INPUT_IMAGE}' in this directory first.")
    
    if not os.path.exists(RAW_GLB):
        step1_crop_portrait(INPUT_IMAGE, CROPPED_IMAGE)
        step2_generate_raw_glb(CROPPED_IMAGE, RAW_GLB)
    
    step3_optimize_and_validate(RAW_GLB, FINAL_GLB)