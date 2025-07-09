import mujoco
import numpy as np
import os
import random
import time
from pathlib import Path
import cv2 # Import OpenCV
import psutil # Import the process utilities library
import gc

def randomize_mesh_scale_with_spec(spec, mesh_name_prefixes):
    """
    Randomly changes the scale of all meshes matching the given prefixes.
    This works even with suffixed names from multiple attachments.
    """
    min_scale_factor = 0.5
    max_scale_factor = 1.5
    scale_factor = np.random.uniform(low=min_scale_factor, high=max_scale_factor)
    print(f"Updating spec for meshes with scale factor: {scale_factor:.3f}")

    # Iterate through all meshes in the spec
    for mesh_spec in spec.meshes:
        if mesh_spec.name: # Check if the mesh has a name
            # Check if the mesh name starts with any of our target prefixes
            for prefix in mesh_name_prefixes:
                if mesh_spec.name.startswith(prefix):
                    # The scale might not be set initially, so we check for None
                    if mesh_spec.scale is None:
                        mesh_spec.scale = np.ones(3)
                    mesh_spec.scale *= scale_factor
                    print(f" - New scale for '{mesh_spec.name}': {np.round(mesh_spec.scale, 3)}")
                    break # Found a match, move to the next mesh

def randomize_vine_position_with_spec(spec, body_name_prefix):
    """
    Randomly changes the position of all bodies matching the given prefix.
    """
    min_pos = [0.0, -1.0, 0.4]
    max_pos = [1.0, 1.0, 0.8]
    
    # Iterate through all bodies in the spec's worldbody
    for body_spec in spec.worldbody.bodies:
        if body_spec.name and body_spec.name.startswith(body_name_prefix):
            new_pos = np.random.uniform(low=min_pos, high=max_pos)
            print(f"Updating spec for body '{body_spec.name}' to position {np.round(new_pos, 3)}")
            body_spec.pos = new_pos

# --- Main Simulation and Viewer Loop (using OpenCV) ---
def main():
    # Get the current process for memory monitoring
    process = psutil.Process(os.getpid())

    # Define file paths
    scene_xml_path = "scene.xml"
    strawb_xml_path = "strawb.xml"
    robot_xml_path = "robot.xml"

    # Ensure all required files exist
    for path in [scene_xml_path, strawb_xml_path, robot_xml_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find required file: {path}")

    # 1. Load the main scene spec
    main_spec = mujoco.MjSpec.from_file(scene_xml_path)

    # 2. Attach multiple strawberry vines to the main scene
    num_vines = 8
    for i in range(num_vines):
        strawb_spec = mujoco.MjSpec.from_file(strawb_xml_path)
        attachment_frame = main_spec.worldbody.add_frame(name=f"vine_mount_{i}")
        
        # Attach the first body from the strawberry spec, adding a suffix
        strawb_root_body = strawb_spec.worldbody.bodies[0]
        attachment_frame.attach_body(strawb_root_body, suffix=f"_{i}")

    # 3. Compile the combined spec into a single model
    model = main_spec.compile()
    data = mujoco.MjData(model)
    
    # Define the prefixes of the meshes and bodies to be randomized
    strawberry_mesh_names = ['strawberry', 'strawberry_leaves', 'strawberry_collision']
    vine_body_prefix = 'vine1'

    # Create the renderer with the new dimensions
    renderer = mujoco.Renderer(model, height=480, width=480)
    
    # Create a camera object and position it to see the scene
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat = [0.5, 0, 0.6]
    cam.distance = 1.0
    cam.elevation = -20.0
    cam.azimuth = 90.0
    
    print("\n" + "="*50)
    print("      >>> IMPORTANT INSTRUCTIONS <<<")
    print("1. An OpenCV window will open.")
    print("2. Press 'S' to randomize the SCALE of all strawberries.")
    print("3. Press 'P' to randomize the POSITION of all strawberries.")
    print("4. Press 'q' or ESC to quit.")
    print("="*50 + "\n")

    try:
        while True:
            step_start = time.time()
            mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
            
            bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            
            # Display memory usage on the frame
            mem_usage_mb = process.memory_info().rss / (1024 * 1024)
            mem_text = f"Memory: {mem_usage_mb:.2f} MB"
            cv2.putText(bgr_pixels, mem_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow("MuJoCo Simulation", bgr_pixels)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                print("-> 's' key detected, randomizing scale...")
                randomize_mesh_scale_with_spec(main_spec, strawberry_mesh_names)
                # Use recompile to update the model and data in-place
                model, data = main_spec.recompile(model, data)
                renderer.close()
                gc.collect()  # Force garbage collection to free up memory
                renderer = mujoco.Renderer(model, height=480, width=480)
                print("   Model recompiled and renderer recreated.")
            elif key == ord('p'):
                print("-> 'p' key detected, randomizing position...")
                randomize_vine_position_with_spec(main_spec, vine_body_prefix)
                # Use recompile to update the model and data in-place
                model, data = main_spec.recompile(model, data)
                renderer.close()
                gc.collect()
                renderer = mujoco.Renderer(model, height=480, width=480)
                print("   Model recompiled and renderer recreated.")

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    finally:
        # Use the suggested `_mjr_context` to check if the renderer is valid
        if hasattr(renderer, '_mjr_context') and renderer._mjr_context is not None:
             renderer.close()
        cv2.destroyAllWindows()
        print("Window closed. Exiting.")

if __name__ == "__main__":
    # You may need to install psutil: pip install psutil
    main()
