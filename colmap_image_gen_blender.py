import bpy, bmesh
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Clear existing mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Specify the path to your OBJ file
obj_file_path = "/run/media/insane/My 4TB 2/Big Data/MPSE/Data/ShapeNetCore/ShapeNetCore.v2/03001627/bf91d0169eae3bfdd810b14a81e12eca/models/model_normalized.obj"


# Import the OBJ file
bpy.ops.import_scene.obj(filepath=obj_file_path)

# Select the imported object
obj = bpy.context.selected_objects[0]

# Scale the object
scale_factor = 3.0  # Scale by a factor of 2
obj.scale = (scale_factor, scale_factor, scale_factor)


# Set the number of images and rotation angles
num_images = 50
num_heights = 5
rotation_angles = [(r*90/num_heights) for r in range(num_heights)]
# Set the background color (RGB values from 0 to 1)
background_color = (0, 0, 0)

USE_RAYTRACING = False

if USE_RAYTRACING:
    # Set render engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'

    # Enable ray tracing
    bpy.context.scene.cycles.use_raytrace = True

# Get the current scene
scene = bpy.context.scene

# Set the background to a solid color
world = scene.world
world.use_nodes = False  # Disable world nodes
world.use_fake_user = True  # Make sure the world is saved

world.color = background_color

# Get the active camera
camera = bpy.data.objects.get("Camera")

print(f"focal length: {camera.data.lens}")
print(f"sensor: {camera.data.sensor_width}, {camera.data.sensor_height}")

# Set the camera as the active object
bpy.context.view_layer.objects.active = camera

# Set the object as active and select it
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# Set camera location (distance from object)
camera.location = (10, 0, 2)  # Adjust the distance and height as needed
# Set the camera to look at the target object
constraint = camera.constraints.new(type='TRACK_TO')
constraint.target = obj
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

cols = []
for i in range(10):
    cols.append(plt.cm.tab10(i/10))

# Create a new material
def create_random_material():
    material = bpy.data.materials.new(name="RandomMaterial")
    material.diffuse_color = random.choice(cols)  # Random RGB color
    return material


bpy.ops.object.mode_set(mode = 'EDIT')          # Go to edit mode to create bmesh
 
bm = bmesh.from_edit_mesh(obj.data)              # Create bmesh object from object mesh

mats = []
for face in bm.faces:
    mats.append(create_random_material())
bpy.ops.object.mode_set(mode = 'OBJECT')

for mat in mats:
    obj.data.materials.append(mat)
    
bpy.ops.object.mode_set(mode = 'EDIT')          # Go to edit mode to create bmesh
 
bm = bmesh.from_edit_mesh(obj.data)              # Create bmesh object from object mesh

for i, face in enumerate(bm.faces):        # Iterate over all of the object's faces
    face.material_index = i

obj.data.update()                            # Update the mesh from the bmesh data
bpy.ops.object.mode_set(mode = 'OBJECT')    # Return to object mode</pre>



# Loop through each rotation angle
for i in tqdm(range(num_images)):
    # Loop through each image
    for angle in rotation_angles:
#        obj.rotation_euler = (
#            math.radians(90), 
#            math.radians(angle), 
#            math.radians(i * 360 / num_images)
#        )
        obj.rotation_euler = (
            math.radians(random.uniform(0, 180)), 
            math.radians(random.uniform(0, 180)), 
            math.radians(random.uniform(0, 180))
        )
        # Render the image
        print(f"Writing {angle}:{i}")
        # Set the output path
        bpy.context.scene.render.filepath = f"/home/insane/blender/{angle}:{i}"
        bpy.ops.render.render(write_still=True)
