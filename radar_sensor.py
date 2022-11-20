import carla 
import math 
import random 
import time 
import numpy as np
import cv2
import open3d as o3d
from matplotlib import cm

#Set up the client using the CARLA client object
client = carla.Client('localhost', 2000)
client.reload_world()
#client.set_timeout(20.0) #seconds
world = client.get_world()

bp_lib = world.get_blueprint_library() 
spawn_points = world.get_map().get_spawn_points() 

# Add vehicle
vehicle_bp = bp_lib.find('vehicle.tesla.model3') 
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[79])

# Move spectator to view ego vehicle
spectator = world.get_spectator()
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)), vehicle.get_transform().rotation)
actor = world.spawn_actor(vehicle_bp, transform)
spectator.set_transform(transform)

# Add traffic and set in motion with Traffic Manager
for i in range(100): 
    vehicle_bp = random.choice(bp_lib.filter('vehicle')) 
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))    
for v in world.get_actors().filter('*vehicle*'): 
    v.set_autopilot(True)
    
# Auxilliary code for colormaps and axes
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:,:3]

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

#Lidar callback

def radar_callback(data, point_list):
    radar_data = np.zeros((len(data), 4))

    for i, detection in enumerate(data):
        x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth) #Defined acc. to the position of the radar
        y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
        z = detection.depth * math.sin(detection.altitude)

        radar_data[i, :] = [x, y, z, detection.velocity] #Velocity towards or away from the detector

    intensity = np.abs(radar_data[:, -1])
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 2])]

    points = radar_data[:, :-1]
    points[:, :1] = -points[:, :1]
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)

def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

#Setting up radar and camera sensors
radar_bp = world.get_blueprint_library().find('sensor.other.radar')
radar_bp.set_attribute('horizontal_fov', '35.0')
radar_bp.set_attribute('vertical_fov', '20.0')
radar_bp.set_attribute('points_per_second', '10000')
radar_init_trans = carla.Transform(carla.Location(x=-2, z=2))
radar = world.spawn_actor(radar_bp, radar_init_trans, attach_to=vehicle)

camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=2.5, z=3), carla.Rotation())
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Add auxilliary data structures
point_list = o3d.geometry.PointCloud()
radar_list = o3d.geometry.PointCloud()

image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

camera_data = {'image': np.zeros((image_h, image_w, 4))}

radar.listen(lambda data: radar_callback(data, radar_list))
camera.listen(lambda image: camera_callback(image, camera_data))

cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
cv2.imshow('RGB Camera', camera_data['image'])
cv2.waitKey(1)

# Open3D visualiser for RADAR
vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name='Carla Radar',
    width=1920,
    height=1080,
    left=960,
    top=540)
vis.get_render_option().background_color = [0.05, 0.05, 0.05]
vis.get_render_option().point_size = 2
vis.get_render_option().show_coordinate_frame = True
add_open3d_axis(vis)

# Update geometry and camera in game loop
frame = 0
while True:
    if frame == 2:
        vis.add_geometry(point_list)
        vis.add_geometry(radar_list)
    vis.update_geometry(point_list)
    vis.update_geometry(radar_list)
    
    vis.poll_events()
    vis.update_renderer()
    # # This can fix Open3D jittering issues:
    time.sleep(0.005)
    frame += 1

    cv2.imshow('RGB Camera', camera_data['image'])
    
    # Break if user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break
        
# Close display and stop sensors
cv2.destroyAllWindows()
radar.stop()
radar.destroy()
camera.stop()
camera.destroy()
vis.destroy_window()

for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for actor in world.get_actors().filter('*sensor*'):
    actor.destroy()







