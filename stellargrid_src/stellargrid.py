import numpy as np
import pandas as pd
import random

class StellarGrid:

    def __init__(self, *args, **kwargs):
        self.star_dict = kwargs.pop('star_dict')
        self.rstar = self.star_dict['rstar']
        self.tile_width = self.star_dict['tile_width'] #units of Rsun
        self.subtile_width = self.star_dict['subtile_width'] 
        self.inc = (self.star_dict['inc'])*np.pi/180  #passed in degrees, convert to radians

        #Differential rotation coefficients in degrees per day
        self.A = self.star_dict['A'] #equatorial velocity
        self.B = self.star_dict['B']
        self.C = self.star_dict['C']  

        self.total_tile_num = 0
        self.visible_full_tiles = 0
            

    @property
    def latitude_num(self):
        """
        Given the stellar grid properties (stellar radius and the tile_width),
        calculate the number of latitudes.
        :return: int
        Number of latitudes in the stellar grid.
        """
        return int(np.round((np.pi * 2 * self.rstar) / (2 * self.tile_width)))

    @property
    def latitude_del(self):
        """
        Calculate the spacing in radians between each latitude.
        :return: float
        Spacing between latitudes in radians.
        """
        return np.pi / ((self.latitude_num) - 1)

    @property
    def latitude_array(self):
        """
        Get an array value of latitude values in the grid (in radians).
        :return: array_like
        Array of latitude values.
        """
        return np.linspace(-np.pi/2, np.pi/2, self.latitude_num)

    def tile_num(self, phi):
        """
        Calculate and return the number of tiles at a given latitude phi (radians).
        :param phi: Latitude (in radians), goes from 0 to pi. Latitude is the angle with respect to the vertical axis Z. When phi = 0, you are at the north pole.
        :return: int
        Number of tiles at a given latitude.
        """
        return int( abs(np.round((2 * np.pi * self.rstar * np.cos(phi)) / self.tile_width)) )

    def theta(self, phi, tile_index):
        """
        For a given latitude and a tile index, compute the theta (longitude) value of the tile to which the index
        corresponds to.
        :param tile_index: Index of the tile (0 to tile_num-1)
        :param phi: Latitude at which the tile is.
        :return: float
        theta (longitude value for the tile) in radians.
        """
        return (2 * np.pi / self.tile_num(phi)) * tile_index
    
    def random_array(self, arr):

        """
        For a given array, shuffle order to begin at a random point 
        """

        array = []

        if len(arr) <= 1:
            return []

        else:
            start_index = random.randint(0, len(arr) - 1)  # Generate a random starting index
            index = start_index
            num_iterations = 0

            while num_iterations < len(arr):
                index = (index + 1) % len(arr)  # Move to the next index in a circular manner
                num_iterations += 1
                array.append(arr[index])

            return array
        
    def shoelace_area(self, corners):
        """
        Calculates the area of a polygon using shoelace theorem (for the poles)
        """
        n = len(corners)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]

        area = abs(area) / 2.0
        return area

    def get_stellar_axis_position(self, phi=None, theta=None):
        """
        For a given phi and theta value, calculate the Cartesian coordinates of a point on the stellar surface
        with respect to a frame fixed to the spherical center of the star. Handedness: X x Y = Z, Z is the rotational axis.
        phi is the latitude, theta is the longitude.
        :param phi: Latitude (radians).
        :param theta Longitude (radians).
        :return: tuple
        Cartesian coordinates of the point on the stellar axis coordinate.
        """

        # x = self.rstar * np.cos(phi) * np.sin(theta)
        # y = self.rstar * np.cos(phi) * np.cos(theta)
        # z = self.rstar * np.sin(phi)

        x = self.rstar * np.cos(phi) * np.sin(theta)
        y = self.rstar * np.sin(phi)
        z = self.rstar * np.cos(phi) * np.cos(theta)

        return np.array([x, y, z])
    
    def get_sky_axis_position(self, phi=None, theta=None):
        """
        For a given phi and theta on the stellar axis, and the user defined stellar inclination
        (with respect to the plane of the sky) and stellar obliquity (with respect to the orbital
        plane of the planet, apply rotations (inclination and obliquity) to the stellar axis
        and then find the Cartesian coordinates of each tile. By default, the sky plane is XZ plane and the observer is located along the +ve Y axis.
        :param phi: Latitude (radians).
        :param theta: Longitude (radians).
        :return: tuple
        Cartesian coordinates of the point on the stellar axis inclined and obliqued.
        """

        xs, ys, zs = self.get_stellar_axis_position(phi=phi, theta=theta)

        x = xs
        y = ys * np.sin(self.inc) - zs * np.cos(self.inc)
        z = ys * np.cos(self.inc) + zs * np.sin(self.inc)

        return np.array([x, y, z])
    

    def get_area_factor_partial_tiles(self, phi_c = None, theta_c = None, phi_low = None, phi_high = None, theta_right = None, theta_left = None):

        """
        For a given partial tile, returns the proportion of tile area that is visible 

        params phi_c & theta_c: spherical coordinates of partial tile center 
        params phi_low & phi_high & theta_right & theta_left: coordinates of partial tile edges - range over which to subtile
        """

        print(phi_c, theta_c, phi_low, phi_high, theta_right, theta_left)

        #number of subtiles in a latitudinal slice
        subtile_num_slice = int(abs(np.round((2 * np.pi * self.rstar * np.cos(phi_c)) / self.subtile_width)) )

        print('Subtile num slice: ', subtile_num_slice)
        
        #number of subtiles in a tile on a given latitudinal slice
        subtile_num_tile = int(subtile_num_slice / self.tile_num(phi_c))
        print('Subtile num tile: ', subtile_num_tile)

        latitude_num = subtile_num_tile #int(abs(phi_low - phi_high) / self.subtile_width)

        latitude_array = np.linspace(phi_low, phi_high, latitude_num)

        visible = 0
        total = 0 

        for i in range(latitude_num):
            phi = latitude_array[i]

            del_theta = abs(theta_left - theta_right) / subtile_num_tile

            for tile_index in range(subtile_num_tile):
                subtile_theta = theta_left + (del_theta * tile_index)

                coords_sky_c = self.get_sky_axis_position(phi=phi, theta=subtile_theta)

                if coords_sky_c[1] > 0:
                    visible += 1
                
                total += 1
        
        return visible/total

    def get_angle_2vector(self, V1=None, V2=None):
        """
        Get the angle in radians between 2 vectors.
        :param V1: array_like
        Vector 1 cartesian coordinates, [V1x, V1y, V1z]

        :param V2: array_like
        Vector 2 cartesian coordinates, , [V2x, V2y, V2z]
        :return: Angle in radians between the V1 and V2 vector.
        """
        V1 = np.array(V1)
        V2 = np.array(V2)
        dot_prod = np.dot(V1, V2)
        scal_prod = np.sqrt(np.sum(V1 ** 2.)) * np.sqrt(np.sum(V2 ** 2.))
        return np.arccos(dot_prod / scal_prod)
    
    def get_vel_rot(self, phi=None):
        """
        Get the velocity of a point on the stellar surface given the differential rotation coefficients
        and the latitude of the point.
        :param A: float
        Equatorial angular velocity in degrees per day.
        :param B: float
        Differential rotation coefficient in degrees per day.
        :param C: float
        Differential rotation coefficient in degrees per day.
        :return: float
        Velocity of the point on the stellar surface in m/s.
        """
        A = self.A
        B = self.B
        C = self.C

        #rotational velocity in degrees per day
        omega_degdays = A + B * np.sin(phi) ** 2 + C * np.sin(phi) ** 4
        #convert to radians per second
        omega_radsec = omega_degdays * (2 * np.pi / 360) * (1 / 86400)
        #convert to linear velocity in m/s
        #v_rot = omega_radsec * (self.rstar * con.R_sun.to(un.m).value)  * np.cos(phi) #m/s

        v_rot = omega_radsec * (self.rstar * 10**6 * np.cos(phi)) #m/s

        return v_rot  #linear velocity of the tile at this latitude
    

    def get_stellar_tile_grid_snapshot(self, rot_ang=None, partial=1):
        
        """
        For the user defined stellar grid properties, and a given rotation angle (0 to pi), get the grid of tiles
        visible to the observer.
        :param rot_ang: stellar rotation angle.
        :return: Dictionary of the tile center, edge coordinates,
        spherical coordinates of the tiles, area of the tiles,
        projected area of the tiles, LOS velocity of the tile,
        Limb darkening of tiles, intensity of tiles.
        """

        tc_star_df, tc_sky_df = [], []
        t1_star_df, t1_sky_df = [], []
        t2_star_df, t2_sky_df = [], []
        t3_star_df, t3_sky_df = [], []
        t4_star_df, t4_sky_df = [], []

        tiles_all = [] # 3D coordinates of all 4 corners of a tile
        tiles_sky = [] # Tile corners completely visible to the sky
        tiles_sky_complete = [] # Complete tiles with all 4 corners visible to the sky
        tiles_sky_complete_center = []
        
        phi_all = []
        theta_all = {}  # Keys are the phi values. For each phi value, store the available theta values.

        # Same as above, but as dataframes (polar coordinates), and only for the center of each tile.
        phi_df = []
        theta_df = []

        # Tile area
        area_df = []
        proj_area_df = []

        lin_vels = []  # Linear velocities for each tile

        # LOS velocities for each tile
        los_vel_sky = []

        area_factor_df = []

        hc_df = []

        north_pole_corners = []
        south_pole_corners = []

        for i in range(self.latitude_num):  # Looping over each latitude.

            phi = self.latitude_array[i]
            phi_all.append(phi)

            del_theta = 2. * np.pi / (
                    self.tile_num(phi) - 1)  # Longitudinal spacing between each tile for this latitude

            # Define latitude of the top and bottom corners of the tile (same for tiles for all longitude at a given
            # latitude)
            phi1 = phi2 = phi - (self.latitude_del / 2)
            phi3 = phi4 = phi + (self.latitude_del / 2)

            # Area of slice of sphere at this latitude
            slice_area = 2. * np.pi * self.rstar ** 2 * (np.sin(phi + self.latitude_del/2) - np.sin(phi - self.latitude_del/2))
            # tile area is slice area/number of tiles on the slice, same for each tile on a latitudinal slice

            if self.tile_num(phi) == 1:
                tile_area = slice_area
            else:
                tile_area = slice_area / self.tile_num(phi)

            theta_all[phi] = []  # Initiate an empty list in the theta dict for this latitude.

            self.total_tile_num+=(self.tile_num(phi))

            arr = np.linspace(0, self.tile_num(phi)-1, self.tile_num(phi))

            #for tile_index in self.random_array(arr): #Looping over each tile starting at a random point (messes with app keeping track of tile number)
            for tile_index in arr:

                tile_theta = self.theta(phi, tile_index) 
                theta_all[phi].append(tile_theta)  # Populate the list of theta for this latitude.

                # Define the longitudes of the left and right edges of the tile at this longitude
                theta1 = theta3 = tile_theta + (del_theta / 2)
                theta2 = theta4 = tile_theta - (del_theta / 2)

                # Save the polar coordinates
                phi_df.append(phi)
                theta_df.append(tile_theta)
                ###############################
                # Get the position of the center of the tile on the stellar axis position
                # phi_theta = [(p, t) for p, t in
                #              zip([phi, phi1, phi2, phi3, phi4], [tile_theta, theta1, theta2, theta3, theta4])]

                # Save the Cartesian coordinates, for the center and all four corners.
                # for k in range(5):
                #     xs, ys, zs = self.get_stellar_axis_position(phi=phi_theta[k][0], theta=phi_theta[k][1])
                #     t_star[k].append([xs, ys, zs])
                #
                #     x, y, z = self.get_sky_axis_position(phi=phi_theta[k][0], theta=phi_theta[k][1])
                #     t_sky[k].append([x, y, z])
                ################################
                ######## Stellar axis position ##########
                coords_star_c = self.get_stellar_axis_position(phi=phi, theta=tile_theta)
                coords_star_1 = self.get_stellar_axis_position(phi=phi1, theta=theta1)
                coords_star_2 = self.get_stellar_axis_position(phi=phi2, theta=theta2)
                coords_star_3 = self.get_stellar_axis_position(phi=phi3, theta=theta3)
                coords_star_4 = self.get_stellar_axis_position(phi=phi4, theta=theta4)
                tc_star_df.append(coords_star_c)
                t1_star_df.append(coords_star_1)
                t2_star_df.append(coords_star_2)
                t3_star_df.append(coords_star_3)
                t4_star_df.append(coords_star_4)

                ######## Sky axis position ##########
                coords_sky_c = self.get_sky_axis_position(phi=phi, theta=tile_theta)
                coords_sky_1= self.get_sky_axis_position(phi=phi1, theta=theta1)
                coords_sky_2 = self.get_sky_axis_position(phi=phi2, theta=theta2)
                coords_sky_3 = self.get_sky_axis_position(phi=phi3, theta=theta3)
                coords_sky_4 = self.get_sky_axis_position(phi=phi4, theta=theta4)

                tc_sky_df.append(coords_sky_c)
                t1_sky_df.append(coords_sky_1)
                t2_sky_df.append(coords_sky_2)
                t3_sky_df.append(coords_sky_3)
                t4_sky_df.append(coords_sky_4)

                if i == 1:
                    north_pole_corners.append([coords_sky_1[0], coords_sky_1[2]])
                
                if i == self.latitude_num - 2:
                    south_pole_corners.append([coords_sky_3[0], coords_sky_3[2]])

                # Save the coordinates of the 4 corners of this tile
                tile_corners = np.array([coords_sky_1, coords_sky_2, coords_sky_3, coords_sky_4])
                corners_z = np.array([coords_sky_1[2], coords_sky_2[2], coords_sky_3[2], coords_sky_4[2]])
                tiles_all.append([coords_sky_1, coords_sky_2, coords_sky_3, coords_sky_4])

                tiles_sky_complete.append(tile_corners)
                tiles_sky_complete_center.append(coords_sky_c)

                if any(corn > 0 for corn in corners_z) == True:  
                    
                    if all(corn > 0 for corn in corners_z) == True: # all 4 corners are visible 
                        area_factor = 1
                        self.visible_full_tiles+=1

                    else:
                        area_factor = float(partial)*self.get_area_factor_partial_tiles(phi, tile_theta, phi1, phi3, theta1, theta2)

                else:
                    area_factor = 0
                       
                area_factor_df.append(area_factor)
                tiles_sky.extend(tile_corners)

                ## Calculate los and then inclined projection 
                j_dir = np.sin(tile_theta)*np.sin(self.inc)
                #j_dir = coords_sky_c[2]/np.sqrt(coords_sky_c[0]**2 + coords_sky_c[1]**2 + coords_sky_c[2]**2) # line of sight direction
                v_diff = self.get_vel_rot(phi) # differential rotation velocity at this latitude
                los_vel = j_dir * v_diff #velocity directed along z axis - line of sight velocity
                

                lin_vels.append(v_diff) # linear velocity of the tile at this latitude
                los_vel_sky.append(np.round(los_vel,8))

                proj_area = area_factor * tile_area * np.sqrt(1 - (coords_sky_c[0]/self.rstar)**2 - (coords_sky_c[1]/self.rstar)**2)
                #print(len(corners_y_plus), coords_sky_c[0], coords_sky_c[2], proj_area)
                proj_area_df.append(proj_area)
                area_df.append(tile_area)     

                heliocentric_angle = self.get_angle_2vector(V1=coords_sky_c, V2=[0,0,1])
                hc_df.append(heliocentric_angle*(180/np.pi)) # convert to degrees


        # ### convert all lists to arrays first before saving in data frame
        tiles_all = np.array(tiles_all) 
        tiles_sky = np.array(tiles_sky)
        tiles_sky_complete = np.array(tiles_sky_complete)
        proj_area_df = np.array(proj_area_df)
        area_df = np.array(area_df)
        area_factor_df = np.array(area_factor_df)

        north_pole_area = self.shoelace_area(north_pole_corners)
        south_pole_area = self.shoelace_area(south_pole_corners)

        if self.inc < np.pi/2 or self.inc > 3*np.pi/2:
            pole_area = north_pole_area
        else:
            pole_area = south_pole_area

        #create a dataframe of theta and phi values

        sky_x1 = np.array([tile[0][0] for tile in tiles_sky_complete])
        sky_x2 = np.array([tile[1][0] for tile in tiles_sky_complete])
        sky_x3 = np.array([tile[2][0] for tile in tiles_sky_complete])
        sky_x4 = np.array([tile[3][0] for tile in tiles_sky_complete])

        sky_y1 = np.array([tile[0][1] for tile in tiles_sky_complete])
        sky_y2 = np.array([tile[1][1] for tile in tiles_sky_complete])
        sky_y3 = np.array([tile[2][1] for tile in tiles_sky_complete])
        sky_y4 = np.array([tile[3][1] for tile in tiles_sky_complete])

        tiles_df = pd.DataFrame({'phi': phi_df, 'theta': theta_df, 'x1': sky_x1, 'y1': sky_y1, 'x2': sky_x2, 'y2': sky_y2, 'x3': sky_x3, 'y3': sky_y3, 'x4': sky_x4, 'y4': sky_y4,
                                 
                                 'angle': hc_df, 'proj_area': proj_area_df, 'lin_vel': lin_vels, 'los_vel': los_vel_sky})

        #dictionary version for muti-axis results
        tiles_dd = {
            'los_vel': los_vel_sky,
            'tiles_sky': tiles_sky,
            'tiles_sky_complete': tiles_sky_complete,
            'area_factor': area_factor_df,
            'proj_area': proj_area_df,
        }

        return tiles_df, tiles_dd 
