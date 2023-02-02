import carla
import glob
import sys
import os
import random
import cv2
import math
import time
import queue
import csv
import open3d as o3d
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from  matplotlib import animation
import pandas as pd

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
inc=0
inc1=0

BB_COLOR = (248, 64, 24)

# Auxilliary code for colormaps and axes
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
print(VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:,:3]

class BoundingBoxGenerator():
    def build_projection_matrix(w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # Now we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth component also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

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
        self.mph = 0

        self.display = None
        self.image = None
        self.capture = True
        self.tgt_vel = 0
    
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
        #radar_bp.set_attribute('sensor_tick', '0.5')
        radar_bp.set_attribute('range', str(60))
        radar_bp.set_attribute('points_per_second', '20000')
        return radar_bp

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
        radar_init_trans = carla.Transform(carla.Location(x=2, z=0.4))
        self.radar = self.world.spawn_actor(self.radar_blueprint(), radar_init_trans, attach_to=vehicle)

    @staticmethod
    def camera_callback(image, data_dict):
        data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    def update_vehvelocity(self, velocity):
        self.vel_x = velocity.x
        self.vel_y = velocity.y
        self.vel_z = velocity.z
        self.mph = int(math.sqrt(self.vel_x**2 + self.vel_y**2 + self.vel_z**2))

    def set_vehstatus(self, status):
        self.status = status
    
    def get_vehstatus(self):
        return self.status
    
    def get_vehvelocity(self):
        return self.mph

    def update_vehtype(self, veh_type):
       self.vehtype = veh_type

    def set_trgt_veh_vel(self, vel):
        self.tgt_vel = vel
    
    def get_trgt_veh_vel(self):
        return self.tgt_vel

    # Radar callback 
    def radar_callback(self, data, point_list):
        radar_data = np.zeros((len(data), 4))
        velocity_range = 7.5 # m/s
        current_rot = data.transform.rotation

        #each radar detection is defined by the depth, altitude and azimuth acc. to the position of the radar        
        filename = '2d_spectrum.csv'
        header = ['det_count', 'depth']
        
        # with open(filename, 'a', newline="") as file:
        #     csvwriter = csv.writer(file)
        #     csvwriter.writerow(header)

        # data_det = [[data.get_detection_count()]]

        # with open(filename, 'a', newline="") as file:
        #     csvwriter = csv.writer(file) 
        #     csvwriter.writerows(data_det)            

        for i, detection in enumerate(data):            
            # inc1+=1
            #print('inc1 ' + str(inc1))
            #print(detection)
            azi = math.degrees(detection.azimuth)
            alt = math.degrees(detection.altitude)
            
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detection.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detection.velocity / velocity_range # range [-1, 1]
            if(norm_velocity > 0):
                r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
                g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
                b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            
                self.world.debug.draw_point(
                    data.transform.location + fw_vec,
                    size=0.075,
                    life_time=0.06,
                    persistent_lines=True,
                    color=carla.Color(r, g, b))

            x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth) #Defined acc. to the position of the radar
            y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
            z = detection.depth * math.sin(detection.altitude) # calculates the height wrt the distance of the object from the sensor and the altitude of the target

            # vehicles = self.world.get_actors().filter('vehicle.*')
            # print('Number of vehicles: % 8d' % len(vehicles))
            # 
            # data_det = [[detection.depth, data.get_detection_count(), data]]
            # with open(filename, 'a', newline="") as file:
            #     csvwriter = csv.writer(file) # 2. create a csvwriter object
            #     csvwriter.writerows(data_det) # 5. write the rest of the data                

            # print('depth  ' + str(detection.depth), 'Vr ' + str(detection.velocity))
            # print('azi ' + str(detection.azimuth), 'alt ' + str(detection.altitude))

            # To get a numpy [[vel, azimuth, altitude, depth],...[,,,]]:
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (len(data), 4))
            #print(points)
            
            """
            detection.velocity = negative, => Vehicle is APPROACHING
            detection.velocity = positive, => Vehicle is LEAVING
            detection.velocity = zero, => object is STATIONARY 
            """
            ego_vel = self.get_vehvelocity() 
            # target_vel = self.get_trgt_veh_vel() 

            #Vr = ego_vel - target_vel
  
            delta = (ego_vel * math.cos(detection.azimuth)) - (abs(detection.velocity)) 
            # print(ego_vel, math.cos(detection.azimuth))
            # print('delta ' + str(int(delta)))
            # print('det_vel ' + str(detection.velocity))         

            # filtering dynamic and ground clutter
            if(z>=0 and z<=3 and (abs(delta)) or ((z>0 and z<2) and (ego_vel==0) and (abs(int(delta))==0))):
                radar_data[i, :] = [x, y, z, (int(delta))] #Velocity towards or away from the detector
                #print('Ramya')
            #print(data.get_detection_count())

            

            #radar_data[i, :] = [x, y, z, abs(delta)<0.1]
                
                # mesh_box = o3d.geometry.TriangleMesh.create_box(width = x, height=y,depth=z)
                # mesh_box.compute_vertex_normals()
                # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                #     size=0.6, origin=[-2, -2, -2])
                # o3d.visualization.draw_geometries(mesh_box, mesh_frame)
                #o3d.visualization.draw_geometries([mesh_box, mesh_frame])
              
            #TBD: Any objects other than vehicles and walkers can be considered as stationary 
    
        intensity = np.abs(radar_data[:, -1])    
        # intensity is a measure of detection.velocity 
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100)) #np.log(np.exp(-0.004 * 100)= -0.4
        
        int_color = np.c_[
            np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
            np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
            np.interp(intensity_col, COOL_RANGE, COOL[:, 2])]
       
        points = radar_data[:, :-1]
        points[:, :1] = -points[:, :1]
        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)
        print(radar_data)
        # if(abs(delta) > 0):            
        #     point_list.colors = o3d.utility.Vector3dVector([0, 0, 0])
        # elif(abs(delta) < 0):
        #     point_list.colors = o3d.utility.Vector3dVector([0, 1, 0])
        # else:
        #     point_list.colors = o3d.utility.Vector3dVector([0, 0, 1])
        
        # print('int color ' + str(int_color))

        data_gen = [data.get_detection_count(), int(detection.depth), int(abs(detection.velocity)), int(abs(delta))]

        with open(filename, 'a', newline="") as file:
            csvwriter = csv.writer(file) 
            csvwriter.writerow(data_gen)
        
        #print('detection.depth ' + str(detection.depth))

        # print(abs(detection.velocity))

        # print(data.get_detection_count())
       
        # # code convert array into list and measure distance
        # L = []
        # pointslist = raw_data_points.tolist()
        # for i in range(len(pointslist)):
        #     L.append(pointslist[i-1][-1])
        
        # ave = sum(L)/len(L)
        #print('Ave ' + str(ave)) #average distance of detected objects

    def game_loop(self):
        """
        Main program loop.
        """
        try:
            #Set up the client using the CARLA client object
            self.client = carla.Client('localhost', 2000)
            self.world = self.client.get_world()
            bp_lib = self.world.get_blueprint_library() 

            # Get the map spawn points
            spawn_points = self.world.get_map().get_spawn_points() 

            # Add vehicle
            vehicle_bp = bp_lib.find('vehicle.tesla.model3') 
            vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            vehicle_bp.set_attribute('color', '255,0,0') # ego vehicle's color set to Red
            #self.transform = random.choice(self.world.get_map().get_spawn_points())
            
            self.setup_camera(vehicle) 
            self.setup_radar(vehicle)
            vehicle.set_autopilot(True)
 
            # Set up the simulator in synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True # Enables synchronous mode
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings) 

            # Get the map spawn points
            spawn_points = self.world.get_map().get_spawn_points()

            # Add traffic and set in motion with Traffic Manager
            #self.spawn_vehicles_around_ego_vehicles(ego_vehicle=vehicle, radius=100, spawn_points=spawn_points, numbers_of_vehicles=10)

            for i in range(50): 
                vehicle_bp = random.choice(bp_lib.filter('vehicle')) 
                npc = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))    
                if npc:
                    npc.set_autopilot(True)  

            #vehicles = self.world.get_actors().filter('vehicle.*')
                                        

            # Create a queue to store and retrieve the sensor data
            image_queue = queue.Queue()
            self.camera.listen(image_queue.put)

            # Get the world to camera matrix
            world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())

            # Get the attributes from the camera
            image_w = self.camera_blueprint().get_attribute("image_size_x").as_int()
            image_h = self.camera_blueprint().get_attribute("image_size_y").as_int()
            fov = self.camera_blueprint().get_attribute("fov").as_float()
            
            # Calculate the camera projection matrix to project from 3D -> 2D
            K = BoundingBoxGenerator.build_projection_matrix(image_w, image_h, fov)

            # Set up the set of bounding boxes from the level
            # We filter for traffic lights and traffic signs
            bounding_box_set = self.world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
            bounding_box_set.extend(self.world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

            # Remember the edge pairs
            edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

            # Retrieve the first image
            self.world.tick()
            image = image_queue.get()

            # Reshape the raw data into an RGB array
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            
            radar_list = o3d.geometry.PointCloud()
            self.radar.listen(lambda data: self.radar_callback(data, radar_list))
            
            # Display the image in an OpenCV display window
            cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB Camera', img)
            cv2.waitKey(1)            

            # Open3D visualiser for RADAR
            vis = o3d.visualization.Visualizer()

            # TBD: radar_list points to BOUNIDNG BOX conversion using open3D
            
            vis.create_window(
                window_name='Carla Radar',
                width=1920,
                height=1080,
                left=960,
                top=540)
            vis.get_render_option().background_color = [0.51, 0.51, 0.51] #np.array([10, 10, 10], np.float32)
            vis.get_render_option().point_size = 4
            vis.get_render_option().show_coordinate_frame = True  
            open3D_visualizer.add_open3d_axis(vis)  

            # Create a 3D box mesh
            box_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=2.0, depth=3.0)   
            bb = box_mesh.get_oriented_bounding_box()  
            print(bb)   

            # Update geometry and camera in game loop
            frame = 0
            try:
                while True:                               
                    if frame == 2:
                        vis.add_geometry(radar_list)
                    vis.update_geometry(radar_list)

                    vis.poll_events()
                    vis.update_renderer()
                    # This can fix Open3D jittering issues:
                    time.sleep(0.005)
                    frame += 1

                    # Retrieve and reshape the image
                    self.world.tick()
                    image = image_queue.get()

                    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

                    # Get the camera matrix 
                    world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())

                    #TBD: Use carla.VehicleWheelLocation to obtain wheel locations 
                    self.set_vehstatus(False) 

                    v = vehicle.get_velocity()
                    self.update_vehvelocity(v)

                    for npc in self.world.get_actors().filter('*vehicle*'):

                        # Filter out the ego vehicle
                        if npc.id != vehicle.id:  
                                        
                            bb = npc.bounding_box
                            dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                           # print('dist = ' + str(dist))
                            # print('Number of vehicles: % 8d' %  (self.world.get_actors().filter('vehicle.*')))

                            # Retrieve ego vehicle velocity
                            # v = vehicle.get_velocity()
                            # self.update_vehvelocity(v)  
                            npc_vel = npc.get_velocity()                                
                            tgt_vel = int(math.sqrt(npc_vel.x**2 + npc_vel.y**2 + npc_vel.z**2)) 
                            self.set_trgt_veh_vel(tgt_vel)                                                    

                            # Filter for the vehicles within 50m same as radar horizontal fov
                            if dist < 50:
              
                                veh_vel = vehicle.get_velocity() 
                                ego_vel = int(math.sqrt(veh_vel.x**2 + veh_vel.y**2 + veh_vel.z**2))

                                rel_vel = ego_vel - tgt_vel
                            
                            # Calculate the dot product between the forward vector
                            # of the vehicle and the vector between the vehicle
                            # and the other vehicle. We threshold this dot product
                            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                                forward_vec = vehicle.get_transform().get_forward_vector()
                                ray = npc.get_transform().location - vehicle.get_transform().location

                                # store the vehicle ids within 35m range
                                self.update_vehtype(npc.type_id) 
                                self.set_vehstatus(True) 
                                # Read the attribute - No. of wheels

                                if forward_vec.dot(ray) > 1:
                                    p1 = BoundingBoxGenerator.get_image_point(bb.location, K, world_2_camera)
                                    verts = [v for v in bb.get_world_vertices(npc.get_transform())] #add bb on the open3d
                                    for edge in edges:
                                        p1 = BoundingBoxGenerator.get_image_point(verts[edge[0]], K, world_2_camera)
                                        p2 = BoundingBoxGenerator.get_image_point(verts[edge[1]],  K, world_2_camera)
                                        cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)

                    cv2.imshow('RGB Camera', img)
                    # Break if user presses 'q'
                    if cv2.waitKey(1) == ord('q'):
                        break   
            except KeyboardInterrupt:
                self.camera.destroy()
                pass
            finally:
                print('Exit')
            vis.destroy_window()       
            for actor in self.world.get_actors().filter('*vehicle*'):
                actor.destroy()
            for actor in self.world.get_actors().filter('*sensor*'):
                actor.destroy()
            for actor in self.world.get_actors().filter('*walker*'):
                actor.destroy()             
        
        finally:
            cv2.destroyAllWindows()  
        vis.destroy_window()   
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