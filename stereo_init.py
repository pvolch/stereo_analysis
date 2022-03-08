import math
import numpy as np
import itertools
from matplotlib.colors import LogNorm
from numpy import cos, sin
import pandas as pd
import sys

def rotate(point, angle, origin = [0,0]):

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def get_rot_axis(a,b,x0,y0,tet):
    a_new = np.tan(np.arctan(a) + tet)
    b_new = y0 - x0*a_new
    #print(a,b,a_new,b_new,tet)
    return(a_new, b_new)

def weight_size(s1, s2, s_sum):
    return (s1 + s2) / s_sum

def weight_sin(phi1, phi2):
    return np.abs(np.sin(phi1 - phi2))

def line_slope_angle(p1,p2):
    return math.atan2(p2[1] - p1[1],p2[0] - p1[0])

def rotate(point, angle, origin = [0,0]):

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def intersect_lines(xp1, yp1, phi1, xp2, yp2, phi2):

    sin_1 = np.sin(phi1)
    cos_1 = np.cos(phi1)
    a1 = sin_1
    b1 = -1 * cos_1
    c1 = yp1 * cos_1 - xp1 * sin_1

    sin_2 = np.sin(phi2)
    cos_2 = np.cos(phi2)

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

    if len(hillas_parameters) < 2:
        return None  # Throw away events with < 2 images

    # Find all pairs of Hillas parameters
    combos = itertools.combinations(list(hillas_parameters), 2)
    hillas_pairs = np.array(list(combos))
    hillas_parameters = np.array(hillas_parameters)
    #Perform intersection
    #xp1, yp1, phi1, xp2, yp2, phi2
    #print(hillas_pairs[:,0,1], hillas_pairs[:,0,2], hillas_pairs[:,0,3], hillas_pairs[:,1,1], hillas_pairs[:,1,2], hillas_pairs[:,1,3])
    sx, sy = intersect_lines(hillas_pairs[:,0,1], hillas_pairs[:,0,2], hillas_pairs[:,0,3], hillas_pairs[:,1,1], hillas_pairs[:,1,2], hillas_pairs[:,1,3])

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

    v = sph2cart(math.radians(90) + azimuth, 0, 1)
    trans = np.array([[(cos(zen) + (1 - cos(zen))*(v[0] ** 2)) , ((1 - cos(zen))*v[0]*v[1] - sin(zen)*v[2]), ((1 - cos(zen))*v[0]*v[2] + sin(zen)*v[1])],
                      [((1 - cos(zen))*v[1]*v[0] + sin(zen)*v[2]), (cos(zen) + (1 - cos(zen))*(v[1] ** 2)), ((1 - cos(zen))*v[1]*v[2] - sin(zen)*v[0])],
                      [((1 - cos(zen))*v[2]*v[0] - sin(zen)*v[1]), ((1 - cos(zen))*v[2]*v[1] + sin(zen)*v[0]), (cos(zen) + (1 - cos(zen))*(v[2] ** 2))]])

    return trans

def reconstruct_tilted(hillas_parameters, tel_x, tel_y):

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
        tel_x_pairs[:, 0], tel_y_pairs[:, 0], hillas_pairs[:,0,3], tel_x_pairs[:, 1], tel_y_pairs[:, 1], hillas_pairs[:,1,3]
    )
    # And sin of interception angle
    weight = weight_size(hillas_pairs[:,0,0], hillas_pairs[:,1,0], np.sum(hillas_parameters[:,0]))*weight_sin(hillas_pairs[:,0,3], hillas_pairs[:,1,3])

    # Make weighted average of all possible pairs
    x_pos = np.average(crossing_x, weights=weight)
    y_pos = np.average(crossing_y, weights=weight)
    var_x = np.average((crossing_x - x_pos) ** 2, weights=weight)
    var_y = np.average((crossing_y - y_pos) ** 2, weights=weight)

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

scalewidth = pd.read_csv('/hdd/IACTs/Corsika/readbin/data/new_cone/MAD_params_new.txt', sep="\s+", header=None);
scalewidth.columns = ["d1", "d2", "k_tab", "g_tab", "a_tab", "b_tab"]
print(scalewidth['d2'])

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def li_ma(n_on, n_off, alp = 1/7):
    return np.sqrt(2)*np.sqrt(n_on*np.log(((1+alp)/alp)*(n_on/(n_on+n_off))) + n_off*np.log((1+alp)*(n_off/(n_on+n_off))))

def get_sb_points(source, r_circle = 0.36):
    antisource = [-source[0], -source[1]]
    source_angle = math.atan2(source[1], source[0])
    antisource_angle = math.atan2(-source[1], -source[0])
    r = np.sqrt(source[0] ** 2 + source[1] ** 2)
    i = 0
    alp = 0
    xy = [list([r*np.cos(source_angle), r*np.sin(source_angle)])]
    while i<4:
        alp = antisource_angle - i*2*np.arccos(((2*(r ** 2)) - (r_circle ** 2))/(2*(r**2)))
        xy.append([r*np.cos(alp), r*np.sin(alp)])
        #print((180/np.pi)*alp)
        if(i!=0):
            alp = antisource_angle + i*2*np.arccos(((2*(r ** 2)) - (r_circle ** 2))/(2*(r**2)))
            xy.append([r*np.cos(alp), r*np.sin(alp)])
        #print((180/np.pi)*alp)
        i+=1
    return xy

def get_alpha(tel, fi_t, tet_t, fi_s, tet_s, x_rot, y_rot, xc, yc, a_axis, b_axis, f, alpha_compare):
    fi_t = math.radians(fi_t) #по азимуту
    tet_t = math.radians(tet_t) #по высоте
    fi_s = math.radians(fi_s)
    tet_s = math.radians(tet_s)
    z = math.sin(tet_s)*math.sin(tet_t) + math.cos(tet_s)*math.cos(tet_t)*math.cos(fi_s - fi_t)
    x0shift = 0.1206*(f/z)*math.cos(tet_s)*math.sin(fi_s - fi_t)
    y0shift = 0.1206*(f/z)*(math.cos(tet_s)*math.sin(tet_t)*math.cos(fi_s - fi_t) - math.sin(tet_s)*math.cos(tet_t))
    alp = np.arctan2(x_rot, y_rot) - np.arctan2(x0shift, y0shift)
    #if(tel == 0):
    #    print(alp*180/np.pi, alpha_compare, abs(alp*180/np.pi + alpha_compare))
    xc_shift, yc_shift = rotate([xc,yc], alp)
    a_new, b_new = get_rot_axis(a_axis, b_axis,xc_shift, yc_shift,alp)
    return (x0shift,y0shift, xc_shift, yc_shift, a_new, b_new)
