import math
import numpy as np
import itertools
from matplotlib.colors import LogNorm
from numpy import cos, sin
import pandas as pd
import sys

def weight_size(s1, s2, s_sum):
    return (s1 + s2) / s_sum

def weight_sin(phi1, phi2):
    return np.abs(np.sin(phi1 - phi2))

def line_slope_angle(p1,p2):
    return math.atan2(p2[1] - p1[1],p2[0] - p1[0])

def rotate(point, angle, origin = [0,0]):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def intersect_lines(xp1, yp1, phi1, xp2, yp2, phi2, azinut):
    """
    Perform intersection of two lines. This code is borrowed from read_hess.

    Parameters
    ----------
    xp1: ndarray
        X position of first image
    yp1: ndarray
        Y position of first image
    phi1: ndarray
        Rotation angle of first image
    xp2: ndarray
        X position of second image
    yp2: ndarray
        Y position of second image
    phi2: ndarray
        Rotation angle of second image

    Returns
    -------
    ndarray of x and y crossing points for all pairs
    """
    sin_1 = np.sin(phi1+azinut)
    cos_1 = np.cos(phi1+azinut)
    a1 = sin_1
    b1 = -1 * cos_1
    c1 = yp1 * cos_1 - xp1 * sin_1

    sin_2 = np.sin(phi2+azinut)
    cos_2 = np.cos(phi2+azinut)

    a2 = sin_2
    b2 = -1 * cos_2
    c2 = yp2 * cos_2 - xp2 * sin_2

    det_ab = a1 * b2 - a2 * b1
    det_bc = b1 * c2 - b2 * c1
    det_ca = c1 * a2 - c2 * a1
    #plt.scatter([xp1, xp2], [yp1, yp2])
    scale = 500
    #plt.plot([xp1-scale*cos_1, xp1+scale*cos_1], [yp1-scale*sin_1, yp1+scale*sin_1])
    #plt.plot([xp2-scale*cos_2, xp2+scale*cos_2], [yp2-scale*sin_2, yp2+scale*sin_2])
    # if  math.fabs(det_ab) < 1e-14 : # /* parallel */
    #    return 0,0
    xs = det_bc / det_ab
    ys = det_ca / det_ab
    return xs, ys

def reconstruct_nominal(hillas_parameters):
    """
    Perform event reconstruction by simple Hillas parameter intersection
    in the nominal system

    Parameters
    ----------
    hillas_parameters: dict
        Hillas parameter objects

    Returns
    -------
    Reconstructed event position in the horizon system
    """
    if len(hillas_parameters) < 2:
        return None  # Throw away events with < 2 images

    # Find all pairs of Hillas parameters
    combos = itertools.combinations(list(hillas_parameters), 2)
    hillas_pairs = np.array(list(combos))
    hillas_parameters = np.array(hillas_parameters)
    #Perform intersection
    #xp1, yp1, phi1, xp2, yp2, phi2
    sx, sy = intersect_lines(hillas_pairs[:,0,1], hillas_pairs[:,0,2], hillas_pairs[:,0,3], hillas_pairs[:,1,1], hillas_pairs[:,1,2], hillas_pairs[:,1,3], 0)

    # Weight by chosen method
    #weight = self._weight_method(h1[3], h2[3])
    # And sin of interception angle
    weight = weight_size(hillas_pairs[:,0,0], hillas_pairs[:,1,0], np.sum(hillas_parameters[:,0]))*weight_sin(hillas_pairs[:,0,3], hillas_pairs[:,1,3])
    # Make weighted average of all possible pairs
    x_pos = np.average(sx, weights=weight)
    y_pos = np.average(sy, weights=weight)
    var_x = np.average((sx - x_pos) ** 2, weights=weight)
    var_y = np.average((sy - y_pos) ** 2, weights=weight)
    return x_pos, y_pos, np.sqrt(var_x), np.sqrt(var_y)

def get_trans_matrix(azimuth, zen):
    """Get Transformation matrix for conversion from the ground system to
    the Tilted system and back again (This function is directly lifted
    from read_hess, probably could be streamlined using python
    functionality)

    Parameters
    ----------
    azimuth: float
        Azimuth angle of the tilted system used
    zen: float
        Zenit angle of the tilted system used

    Returns
    -------
    trans: 3x3 ndarray transformation matrix
    """

    v = sph2cart(math.radians(90) + azimuth, 0, 1)
    trans = np.array([[(cos(zen) + (1 - cos(zen))*(v[0] ** 2)) , ((1 - cos(zen))*v[0]*v[1] - sin(zen)*v[2]), ((1 - cos(zen))*v[0]*v[2] + sin(zen)*v[1])],
                      [((1 - cos(zen))*v[1]*v[0] + sin(zen)*v[2]), (cos(zen) + (1 - cos(zen))*(v[1] ** 2)), ((1 - cos(zen))*v[1]*v[2] - sin(zen)*v[0])],
                      [((1 - cos(zen))*v[2]*v[0] - sin(zen)*v[1]), ((1 - cos(zen))*v[2]*v[1] + sin(zen)*v[0]), (cos(zen) + (1 - cos(zen))*(v[2] ** 2))]])

    return trans

def reconstruct_tilted(hillas_parameters, tel_x, tel_y, azimut, source_x, source_y):
    """
    Core position reconstruction by image axis intersection in the tilted
    system

    Parameters
    ----------
    hillas_parameters: dict
        Hillas parameter objects
    tel_x: dict
        Telescope X positions, tilted system
    tel_y: dict
        Telescope Y positions, tilted system

    Returns
    -------
    (float, float, float, float):
        core position X, core position Y, core uncertainty X,
        core uncertainty X
    """
    est = []
    if len(hillas_parameters) < 2:
        return None  # Throw away events with < 2 images

    # Find all pairs of Hillas parameters
    combos = itertools.combinations(list(hillas_parameters), 2)
    hillas_pairs = np.array(list(combos))
    hillas_parameters = np.array(hillas_parameters)

    # Find all pairs of Hillas parameters
    tel_x_pairs = np.array(list(itertools.combinations(tel_x, 2)))
    tel_y_pairs = np.array(list(itertools.combinations(tel_y, 2)))
    #print(tel_x_pairs, tel_y_pairs, hillas_pairs)
    #print(tel_x_pairs[:, 0], tel_y_pairs[:, 0], hillas_pairs[:,0,3], tel_x_pairs[:, 1], tel_y_pairs[:, 1], hillas_pairs[:,1,3])
    # Perform intersection
    crossing_x, crossing_y = intersect_lines(
        tel_x_pairs[:, 0], tel_y_pairs[:, 0], hillas_pairs[:,0,4], tel_x_pairs[:, 1], tel_y_pairs[:, 1], hillas_pairs[:,1,4], azimut
    )
    for i in range(len(tel_x_pairs)):
        xs, ys = rotate([source_x/100, source_y/100], azimut)
        xc1,yc1 = rotate([hillas_pairs[i,0,1]/100, hillas_pairs[i,0,2]/100], azimut)
        xc2,yc2 = rotate([hillas_pairs[i,1,1]/100, hillas_pairs[i,1,2]/100], azimut)

        unit_vector_1 = [xs-xc1, ys-yc1] / np.linalg.norm([xs-xc1, ys-yc1])
        unit_vector_2 = [crossing_x[i]-tel_x_pairs[i,0], crossing_y[i]-tel_y_pairs[i,0]] / np.linalg.norm([crossing_x[i]-tel_x_pairs[i,0], crossing_y[i]-tel_y_pairs[i,0]])
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        #print(unit_vector_1, unit_vector_2,dot_product)
        angle1 = np.arccos(round(dot_product,2))

        #print('ang1:', angle1)

        unit_vector_1 = [xs-xc2, ys-yc2] / np.linalg.norm([xs-xc2, ys-yc2])
        unit_vector_2 = [crossing_x[i]-tel_x_pairs[i,1], crossing_y[i]-tel_y_pairs[i,1]] / np.linalg.norm([crossing_x[i]-tel_x_pairs[i,1], crossing_y[i]-tel_y_pairs[i,1]])
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle2 = np.arccos(round(dot_product,2))

        #print('ang2:', angle2)

        if(abs(angle1) > np.pi/2 or abs(angle2) > np.pi/2):
                est.append(False)
        else:
            est.append(True)
    # And sin of interception angle
    weight = weight_size(hillas_pairs[:,0,0], hillas_pairs[:,1,0], np.sum(hillas_parameters[:,0]))*weight_sin(hillas_pairs[:,0,4], hillas_pairs[:,1,4])
    if np.sum(est) == 0:
        #print('here!')
        return None, None, None, None
    # Make weighted average of all possible pairs
    x_pos = np.average(crossing_x[est], weights=weight[est])
    y_pos = np.average(crossing_y[est], weights=weight[est])
    var_x = np.average((crossing_x[est] - x_pos) ** 2, weights=weight[est])
    var_y = np.average((crossing_y[est] - y_pos) ** 2, weights=weight[est])

    return x_pos, y_pos, np.sqrt(var_x), np.sqrt(var_y)

def norm_width(scalewidth, hillas, tel_coord, x_ground, y_ground):
    nwidth = 0
    normalize = 0
    tel_coord = np.array(tel_coord)
    hillas = np.array(hillas)
    for j in range(len(hillas)):
        find_nwidth = 0
        distance = math.sqrt(pow(x_ground - tel_coord[j][0],2) + pow(y_ground - tel_coord[j][1],2))
        for i, (d1, d2) in enumerate(zip(scalewidth['d1'],scalewidth['d2'])):
            if(distance >= d1 and distance < d2):
                find_nwidth = 1
                normalize+=1
                size_event = hillas[j,0]
                width_event = hillas[j,5]
                med = scalewidth['k_tab'][i]*pow(size_event, scalewidth['g_tab'][i]);
                MADmed = scalewidth['a_tab'][i]*size_event + scalewidth['b_tab'][i];
                nwidth = nwidth + (width_event - med)/(MADmed)
        if(find_nwidth == 0 and distance >= scalewidth['d2'].iloc[-1]):
            normalize+=1
            size_event = hillas[j,0]
            width_event = hillas[j,5]
            med = scalewidth['k_tab'].iloc[-1]*pow(size_event, scalewidth['g_tab'].iloc[-1]);
            MADmed = scalewidth['a_tab'].iloc[-1]*size_event + scalewidth['b_tab'].iloc[-1];
            nwidth = nwidth + (width_event - med)/(MADmed)
    if(normalize > 0):
        if(abs(nwidth/normalize) < 1000):
            return nwidth/normalize
        else:
            return None
    else:
        return None

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def reconstruct_xmax(source_x, source_y, core_x, core_y, hillas_parameters, x_tel_tilt, y_tel_tilt, zen, r_center):
        #print(source_x, source_y, core_x, core_y, hillas_parameters, tel_coord, alt)
        """
        Geometrical depth of shower maximum reconstruction, assuming the shower
        maximum lies at the image centroid

        Parameters
        ----------
        source_x: float
            Source X position in nominal system
        source_y: float
            Source Y position in nominal system
        core_x: float
            Core X position in nominal system
        core_y: float
            Core Y position in nominal system
        hillas_parameters: dict
            Dictionary of hillas parameters objects
        tel_x: dict
            Dictionary of telescope X positions in tilted frame
        tel_y: dict
            Dictionary of telescope Y positions in tilted frame
        zen: float
            Zenith angle of shower

        Returns
        -------
        float:
            Estimated depth of shower maximum
        """
        hillas_parameters = np.array(hillas_parameters)
        """print(
            source_x,
            source_y,
            hillas_parameters[:,0],
            hillas_parameters[:,1],
            hillas_parameters[:,2],
            core_x,
            core_y,
            x_tel_tilt,
            y_tel_tilt
        )
        """
        height = get_shower_height(
            source_x,
            source_y,
            hillas_parameters[:,1], # 6, 7 - use brightest pixel; 1,2 - use image CoG
            hillas_parameters[:,2],
            core_x,
            core_y,
            x_tel_tilt,
            y_tel_tilt,
        )
        weight = np.array(hillas_parameters[:,0])
        mean_height = np.sum(height * weight) / np.sum(weight)

        # This value is height above telescope in the tilted system,
        # we should convert to height above ground
        #print(mean_height, mean_height*np.cos(zen))
        mean_height *= np.cos(zen)
        # Add on the height of the detector above sea level
        mean_height += 475  # TODO: replace with instrument info

        if mean_height > 100000 or np.isnan(mean_height):
            mean_height = 100000
            return np.nan
        # Lookup this height in the depth tables, the convert Hmax to Xmax
        # x_max = self.thickness_profile(mean_height.to(u.km))
        # Convert to slant depth
        # x_max /= np.cos(zen)
        P0 = 1033
        g = 9.807
        Rconst = 8.31
        M = 28.964
        t11 = -56.5
        t0 = -17.5
        gradT = (t11 -t0)/11
        xmax = (P0 * (1 + gradT*(mean_height/1e3)/(256.15)) ** (-g*M/(gradT*Rconst)))
        #print(mean_height, xmax)
        dxmax = -0.13880345*r_center + 58.15229779
        #dxmax = 0
        return xmax - dxmax

def get_shower_height(source_x, source_y, cog_x, cog_y, core_x, core_y, tel_pos_x, tel_pos_y):
    """
    Function to calculate the depth of shower maximum geometrically under the assumption
    that the shower maximum lies at the brightest point of the camera image.
    Parameters
    ----------
    source_x: float
        Event source position in nominal frame
    source_y: float
        Event source position in nominal frame
    cog_x: list[float]
        Center of gravity x-position for all the telescopes in rad
    cog_y: list[float]
        Center of gravity y-position for all the telescopes in rad
    core_x: float
        Event core position in telescope tilted frame
    core_y: float
        Event core position in telescope tilted frame
    tel_pos_x: list
        List of telescope X positions in tilted frame
    tel_pos_y: list
        List of telescope Y positions in tilted frame

    Returns
    -------
    float: Depth of maximum of air shower
    """

    # Calculate displacement of image centroid from source position (in rad)
    disp = (np.pi/180.0)*np.sqrt((cog_x - source_x) ** 2 + (cog_y - source_y) ** 2)
    #print(disp)
    # Calculate impact parameter of the shower
    impact = np.sqrt((tel_pos_x - core_x) ** 2 + (tel_pos_y - core_y) ** 2)
    #print(impact)
    # Distance above telescope is ration of these two (small angle)
    height = impact / disp

    return height

def get_energy(sizes, Rtel, xmax, coefs_energy, delta_distance, delta_xmax):
    energy = []
    for itel in range(len(sizes)):
        for i in range(len(coefs_energy)):
            if((Rtel[itel] > coefs_energy['distance'][i]) and (Rtel[itel] <= (coefs_energy['distance'][i] + delta_distance)) and (xmax > coefs_energy['xmax'][i]) and (xmax <= (coefs_energy['xmax'][i] + delta_xmax))):
                if(coefs_energy['k_axis'][i] != coefs_energy['b_axis'][i] and coefs_energy['b_axis'][i] !=0):
                    energy.append(coefs_energy['k_axis'][i]*sizes[itel] + coefs_energy['b_axis'][i])
                    break
                else:
                    energy.append(None)
                    break
        if len(energy) < itel+1:
            energy.append(None)
    return(np.array(energy))

def get_tilted_xy(tel_coord, normal, M):
    t = (-normal[0]*tel_coord[0]-normal[1]*tel_coord[1]-normal[2]*tel_coord[2])/(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    x_tilt = M[0][0] * (tel_coord[0]+t*normal[0]) + M[1][0] * (tel_coord[1]+t*normal[1]) + M[2][0] * (tel_coord[2]+t*normal[2])
    y_tilt = M[0][1] * (tel_coord[0]+t*normal[0]) + M[1][1] * (tel_coord[1]+t*normal[1]) + M[2][1] * (tel_coord[2]+t*normal[2])
    return (x_tilt, y_tilt)

def draw_axis(reco_axis, true_axis, tel_coord, hillas, azimuth):
    hillas = np.array(hillas)
    fig, ax = plt.subplots(figsize=(10, 10))
    print(len(tel_coord))
    for i in range(len(tel_coord[0])):
        ax.scatter(tel_coord[0][i], tel_coord[1][i], marker="o", c = 'red', s=50)
        point1 = [-500*np.cos(azimuth+hillas[i,4])+tel_coord[0][i], -500*np.sin(azimuth+hillas[i,4])+tel_coord[1][i]]
        point2 = [500*np.cos(azimuth+hillas[i,4])+tel_coord[0][i], 500*np.sin(azimuth+hillas[i,4])+tel_coord[1][i]]
        ax.plot([point1[0],point2[0]], [point1[1],point2[1]])
    ax.scatter(reco_axis[0], reco_axis[1], c = 'black')
    ax.scatter(true_axis[0], true_axis[1])
    #ax.set_xlim([reco_axis[0]-25, reco_axis[0]+25])
    #ax.set_ylim([reco_axis[1]-25, reco_axis[1]+25])
    plt.show()
    #sys.exit()
