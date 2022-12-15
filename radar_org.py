import carla
import glob
import sys
import os
import random
import cv2
import math
import time
import open3d as o3d
from matplotlib import cm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

# Auxilliary code for colormaps and axes
VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:,:3]

class open3D_visualizer():
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

class BasicSensorClients():

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.capture = True
    
    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def radar_blueprint(self):
        """
        Returns radar blueprint.
        """
        radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        #radar_bp.set_attribute('noise_seed','10')
        radar_bp.set_attribute('horizontal_fov', '35.0')
        radar_bp.set_attribute('vertical_fov', '20.0')
        radar_bp.set_attribute('points_per_second', '10000')
        return radar_bp

    # Radar callback 
    def radar_callback(data, point_list):
        radar_data = np.zeros((len(data), 4))

        #each radar detection is defined by the depth, altitude and azimuth acc. to the position of the radar
        for i, detection in enumerate(data):
            x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth) #Defined acc. to the position of the radar
            y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
            z = detection.depth * math.sin(detection.altitude)

            #also each detection has a velocity towards/away from the detector
            if(z>0):
                radar_data[i, :] = [x, y, z, detection.velocity] #Velocity towards or away from the detector
            #radar_data[i, :] = [x, y, z, detection.velocity]

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

    #Camera callback          
    def setup_camera(self, vehicle):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        camera_bp = self.camera_blueprint()
        camera_transform = carla.Transform(carla.Location(x=-7.5, z=2.8), carla.Rotation(yaw=0, pitch=0, roll=0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    def setup_radar(self, vehicle):
        """
        Spawns actor-radar to be used to .
        Sets calibration for client-side boxes rendering.
        """
        radar_init_trans = carla.Transform(carla.Location(x=-2, z=0.4))
        self.radar = self.world.spawn_actor(self.radar_blueprint(), radar_init_trans, attach_to=vehicle)

    @staticmethod
    def camera_callback(image, data_dict):
        data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    def game_loop(self):
        """
        Main program loop.
        """
        try:
            #Set up the client using the CARLA client object
            self.client = carla.Client('localhost', 2000)
            self.client.reload_world()
            self.world = self.client.get_world()

            bp_lib = self.world.get_blueprint_library() 
            spawn_points = self.world.get_map().get_spawn_points() 

            # Add vehicle
            vehicle_bp = bp_lib.find('vehicle.tesla.model3') 
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[79])

            # # Move spectator to view ego vehicle
            # spectator = self.world.get_spectator()
            # transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)), self.vehicle.get_transform().rotation)
            # actor = self.world.spawn_actor(vehicle_bp, transform)
            # spectator.set_transform(transform)

            # Add traffic and set in motion with Traffic Manager
            for i in range(100): 
                vehicle_bp = random.choice(bp_lib.filter('vehicle')) 
                npc = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))    
            for v in self.world.get_actors().filter('*vehicle*'): 
                v.set_autopilot(True)                   

            image_w = self.camera_blueprint().get_attribute("image_size_x").as_int()
            image_h = self.camera_blueprint().get_attribute("image_size_y").as_int()
            camera_data = {'image': np.zeros((image_h, image_w, 4))}

            self.setup_radar(vehicle)
            self.setup_camera(vehicle) 

            radar_list = o3d.geometry.PointCloud()              

            self.radar.listen(lambda data: BasicSensorClients.radar_callback(data, radar_list))
            self.camera.listen(lambda image: BasicSensorClients.camera_callback(image, camera_data))        

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
            open3D_visualizer.add_open3d_axis(vis)          

            # Update geometry and camera in game loop
            frame = 0
            while True:                               
                if frame == 2:
                    vis.add_geometry(radar_list)
                vis.update_geometry(radar_list)

                vis.poll_events()
                vis.update_renderer()
                # This can fix Open3D jittering issues:
                time.sleep(0.005)
                frame += 1

                cv2.imshow('RGB Camera', camera_data['image'])

                # Break if user presses 'q'
                if cv2.waitKey(1) == ord('q'):
                    break   

            vis.destroy_window()       
            for actor in self.world.get_actors().filter('*vehicle*'):
                actor.destroy()
            for actor in self.world.get_actors().filter('*sensor*'):
                actor.destroy()               

        finally:
            cv2.destroyAllWindows()  
            
# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    """
    Initializes the client-side bounding box demo.
    """
    try:
        client = BasicSensorClients()
        client.game_loop()
    finally:
        print('EXIT')

if __name__ == '__main__':
    main()