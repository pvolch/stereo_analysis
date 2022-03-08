import math
import numpy as np
import itertools
from matplotlib.colors import LogNorm
from numpy import cos, sin
import pandas as pd
import sys
import csv
import stereo_init

focal = [485, 482]
x_tel = [108.5218, -35.51704];
y_tel = [91.30141, -197.1544];
z_tel = [1.01, 1.55]
rd_dep = []
all_time = 0
time0 = 0
nwidth_array = [0,0,0,0,0,0,0,0]
tet2_array = [0,0,0,0,0,0,0,0]
source_distance = []
data_run = ['171020', '181020','201020', '211020', '221020', '221020.01', '231020', '251020', '261020', '271020',
            '101120', '111120', '141120', '161120', '191120', '191120.01', '101220.00', '111220', '111220.02', '151220', '151220.01',
            '171220.00', '181220.00', '191220.00', '201220.00', '221220.00', '090121.00', '090121.01', '120121.00', '130121.00', '140121.00', '160121.00', '180121.00']
#data_run = ['171220.00']
save_file = open('events.stereo_171220', 'w')
write = csv.writer(save_file, escapechar=' ', quoting=csv.QUOTE_NONE)
write.writerow(['run,unix_time,event_IACT01,event_IACT02,size_IACT01,size_IACT02,dist[0]_IACT01,dist[0]_IACT02,CR_stereo,tet2_0,nwidth_0,tet2_1,nwidth_1,tet2_2,nwidth_2,tet2_3,nwidth_3,tet2_4,nwidth_4,tet2_5,nwidth_5,tet2_6,nwidth_6,tet2_7,nwidth_7'])
size_cut = 120
xyc_cut = 3.5
delta_time_cut = 0.5
cr_stereo_cut = 1.0
error_deg_cut = 0.1
xyc_dist_cut = 3.5
source_location_cut = float('inf')
normalized_width_cut = float('inf')
#data_run = ['111220', '111220.02', '151220', '151220.01']
columns = ['tel', 'cr_stereo', 'por', 'event_numb', 'unix_time', 'unix time after dot(ns)', 'delta_time', 'error_deg', 'tel_az', 'tel_el',
           'source_az', 'source_el', 'CR100phe', 'CR_portion', 'numb_pix', 'size', 'Xc[0]','Yc[0]', 'con2', 'length[0]',
           'width[0]', 'dist[0]', 'dist[1]', 'dist[2]', 'azwidth[1]', 'azwidth[2]', 'miss[1]', 'miss[2]', 'alpha[0]',
           'alpha[1]', 'alpha[2]', 'a_axis', 'b_axis', 'a_dist[1]', 'b_dist[1]', 'a_dist[2]', 'b_dist[2]', 'tel_ra',
           'tel_dec', 'source_ra', 'source_dec', 'source_x', 'source_y', 'tracking', 'good', 'star', 'edge', 'weather_mark', 'alpha_c']
location_sum = [0,0,0,0,0,0,0,0]
n_width_sum =  [0,0,0,0,0,0,0,0]
for run in data_run:
    location = [0,0,0,0,0,0,0,0]
    n_nwidth = [0,0,0,0,0,0,0,0]
    #nwidth_array = [[],[]]
    with open('/hdd/IACTs/Crab/2020-21/stereo/data/new/' + run + '.stereo') as file:
        for line in file:
            #print(line)
            xy_source_new = []
            new_param_2tel = []
            param_2tel = []
            stereo_cr = []
            for i in range(int(line.split(',')[1])):
                header = file.readline()
                if(len(header) > 0):
                    #print(run, header.split()[2])
                    dict = {}
                    for j in range(len(header.split(','))):
                        dict[columns[j]] = float(header.split(',')[j])
                    param_2tel.append(dict)
            n_tel = 0
            for i in range(int(line.split(',')[1])):
                xyc_dist0 = np.sqrt(param_2tel[i]['Xc[0]'] ** 2 + param_2tel[i]['Yc[0]'] ** 2)
                #print(param_2tel[i]['event_numb'])
                if(param_2tel[i]['size'] > size_cut and param_2tel[i]['delta_time'] < delta_time_cut and param_2tel[i]['error_deg'] < error_deg_cut and xyc_dist0 < xyc_dist_cut and param_2tel[i]['cr_stereo'] > cr_stereo_cut):
                    #print(param_2tel[i]['cr_stereo'])
                    if(param_2tel[0]['unix_time'] - time0 < 240):
                        all_time = all_time + (param_2tel[0]['unix_time'] - time0)
                    time0 = param_2tel[0]['unix_time']
                    x_source_new, y_source_new, xc_new, yc_new, a_new, b_new = stereo_init.get_alpha(param_2tel[i]['tel'],
                                                                                        param_2tel[i]['tel_az'],
                                                                                        param_2tel[i]['tel_el'],
                                                                                        param_2tel[i]['source_az'],
                                                                                        param_2tel[i]['source_el'],
                                                                                        param_2tel[i]['source_x'],
                                                                                        param_2tel[i]['source_y'],
                                                                                        param_2tel[i]['Xc[0]'],
                                                                                        param_2tel[i]['Yc[0]'],
                                                                                        param_2tel[i]['a_axis'],
                                                                                        param_2tel[i]['b_axis'],
                                                                                        focal[int(param_2tel[i]['tel'])],
                                                                                        param_2tel[i]['alpha_c'])
                    new_param_2tel.append([int(param_2tel[i]['tel']),x_source_new, y_source_new, xc_new, yc_new, a_new, b_new])
                    n_tel+=1
            if(n_tel == 2):
                xyc1_xyc2_dist = np.sqrt((new_param_2tel[0][1] - new_param_2tel[1][1]) ** 2 + (new_param_2tel[0][2] - new_param_2tel[1][2]) ** 2)
                new_source = [new_param_2tel[0][1], new_param_2tel[0][2]]
                r_new_source = np.sqrt(new_source[0] ** 2 + new_source[1] ** 2)
            if(n_tel == 2 and xyc1_xyc2_dist < 0.5 and r_new_source > 1):
                #fig, ax = plt.subplots(1,2, figsize=(12, 6))
                #ax[0].scatter([new_param_2tel[0][1], new_param_2tel[1][1]], [new_param_2tel[0][2], new_param_2tel[1][2]])
                #ax[0].scatter([new_param_2tel[0][3], new_param_2tel[1][3]], [new_param_2tel[0][4], new_param_2tel[1][4]])
                #ax[0].plot([-5,5], [new_param_2tel[0][5]*(-5) + new_param_2tel[0][6], new_param_2tel[0][5]*(5) + new_param_2tel[0][6]])
                #ax[0].plot([-5,5], [new_param_2tel[1][5]*(-5) + new_param_2tel[1][6], new_param_2tel[1][5]*(5) + new_param_2tel[1][6]])
                source_distance.append(np.sqrt(new_source[0] ** 2 + new_source[1] ** 2))
                shift_source = [-new_param_2tel[0][1] + new_param_2tel[1][1], -new_param_2tel[0][2] + new_param_2tel[1][2]]
                new_xyc = [[new_param_2tel[0][3], new_param_2tel[0][4]], [new_param_2tel[1][3]-shift_source[0], new_param_2tel[1][4] - shift_source[1]]]
                #new_source = [new_param_2tel[0][1], new_param_2tel[0][2]]
                new_ab = [[new_param_2tel[0][5], new_param_2tel[0][6]], [new_param_2tel[1][5], new_xyc[1][1] - new_param_2tel[1][5]*new_xyc[1][0]]]
                xy_sb  = stereo_init.get_sb_points(new_source) # 1- source, other - background
                #ax[1].scatter(new_source[0], new_source[1])
                #ax[1].scatter([new_xyc[0][0], new_xyc[1][0]], [new_xyc[0][1], new_xyc[1][1]])
                #ax[1].plot([-5,5], [new_ab[0][0]*(-5) + new_ab[0][1], new_ab[0][0]*(5) + new_ab[0][1]])
                #ax[1].plot([-5,5], [new_ab[1][0]*(-5) + new_ab[1][1], new_ab[1][0]*(5) + new_ab[1][1]])
                #ax[1].set_xlim([-4,0])
                #ax[1].set_ylim([-2,2])
                #sys.exit()
                for imode,xy in enumerate(xy_sb):
                    hillas = []
                    tel_coord = []
                    for i in range(int(line.split(',')[1])):
                        dist_slope = stereo_init.line_slope_angle([new_xyc[i][0],new_xyc[i][1]],[xy[0], xy[1]])
                        hillas.append([param_2tel[i]['size'], new_xyc[i][0], new_xyc[i][1], math.atan(new_ab[i][0]), dist_slope, param_2tel[i]['width[0]']])
                        tel_coord.append([x_tel[int(param_2tel[i]['tel'])], y_tel[int(param_2tel[i]['tel'])], z_tel[int(param_2tel[i]['tel'])]])
                    tel_coord = np.array(tel_coord)
                    reconst = stereo_init.reconstruct_nominal(hillas)
                    x_source_error = reconst[0] - xy[0]
                    y_source_error = reconst[1] - xy[1]
                    #print(hillas)
                    if(math.sqrt(x_source_error ** 2 + y_source_error ** 2) < source_location_cut):
                        location[imode]+=1
                        M = stereo_init.get_trans_matrix(np.pi*param_2tel[0]['tel_az']/180.0, np.pi*(90 - param_2tel[0]['tel_el'])/180.0)
                        normal = stereo_init.sph2cart(np.pi*param_2tel[0]['tel_az']/180.0, np.pi*(param_2tel[0]['tel_el'])/180.0, 1)
                        t = (-normal[0]*tel_coord[:,0]-normal[1]*tel_coord[:,1]-normal[2]*tel_coord[:,2])/(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
                        x_tilt = M[0][0] * (tel_coord[:,0]+t*normal[0]) + M[1][0] * (tel_coord[:,1]+t*normal[1]) + M[2][0] * (tel_coord[:,2]+t*normal[2])
                        y_tilt = M[0][1] * (tel_coord[:,0]+t*normal[0]) + M[1][1] * (tel_coord[:,1]+t*normal[1]) + M[2][1] * (tel_coord[:,2]+t*normal[2])
                        z_tilt = M[0][2] * (tel_coord[:,0]+t*normal[0]) + M[1][2] * (tel_coord[:,1]+t*normal[1]) + M[2][2] * (tel_coord[:,2]+t*normal[2])
                        reconst_titled = stereo_init.reconstruct_tilted(hillas, x_tilt, y_tilt)

                        Rm = np.sqrt((x_tilt - reconst_titled[0]) ** 2 + (y_tilt - reconst_titled[1]) ** 2)
                        rd_dep.append([Rm[0], Rm[1], param_2tel[0]['size'], param_2tel[1]['size']])
                        #print(Rm, param_2tel[0]['size'], param_2tel[1]['size'])

                        x_reco_tilt_g = M[0, 0] * reconst_titled[0] + M[0, 1] * reconst_titled[1]
                        y_reco_tilt_g = M[1, 0] * reconst_titled[0] + M[1, 1] * reconst_titled[1]
                        z_reco_tilt_g = M[2, 0] * reconst_titled[0] + M[2, 1] * reconst_titled[1]

                        x_projected = x_reco_tilt_g + M[2][0] * z_reco_tilt_g / M[2][2]
                        y_projected = y_reco_tilt_g + M[2][1] * z_reco_tilt_g / M[2][2]

                        #reconstruct_xmax(param['x_source']*0.1206, param['y_source']*0.1206, reconst[0], reconst[1], hillas, x_ground, y_ground, param['altitude']) #вместо x_ground y_ground вписать положения телескопов
                        nwidth = stereo_init.norm_width(stereo_init.scalewidth, hillas, tel_coord, x_projected, y_projected)
                        if nwidth != None:
                            nwidth_array[imode] = nwidth
                        if nwidth != None and nwidth < normalized_width_cut:
                            tet2_array[imode] = math.sqrt(x_source_error ** 2 + y_source_error ** 2)
                            n_nwidth[imode]+=1
                            #print(mode, nwidth, math.sqrt(x_source_error ** 2 + y_source_error ** 2), datetime.fromtimestamp(param_2tel[0]['unix_time']))
                            #print(mode, param_2tel[0]['por'], datetime.fromtimestamp(param_2tel[0]['unix_time']))
                save_list = [run, param_2tel[0]['unix_time'], param_2tel[0]['event_numb'], param_2tel[1]['event_numb'], round(param_2tel[0]['size'],2), round(param_2tel[1]['size'],2),
                                round(np.sqrt(param_2tel[0]['Xc[0]'] ** 2 + param_2tel[0]['Yc[0]'] ** 2),3), round(np.sqrt(param_2tel[1]['Xc[0]'] ** 2 + param_2tel[1]['Yc[0]'] ** 2),3),
                                round(param_2tel[1]['cr_stereo'],1)]
                for imode,xy in enumerate(xy_sb):
                    save_list.append(round(tet2_array[imode],3))
                    save_list.append(round(nwidth_array[imode],2))
                write.writerow(save_list)
        print(run, location, n_nwidth)        #sys.exit(0)
        for i in range(len(xy_sb)):
            location_sum[i] = location_sum[i] + location[i]
            n_width_sum[i] = n_width_sum[i] + n_nwidth[i]
print(location_sum, n_width_sum)
print(stereo_init.li_ma(location_sum[0], sum(location_sum[1:]), alp = 1/7), stereo_init.li_ma(n_width_sum[0], sum(n_width_sum[1:]), alp = 1/7))
print(all_time/(60*60))
