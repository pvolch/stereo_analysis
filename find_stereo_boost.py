import numpy as np
import pandas as pd
import copy
import glob
import math
from numba import jit
from astropy.time import Time
import matplotlib.pyplot as plt
import time
import sys
from datetime import datetime
start_time = time.time()

def output_element(number, element):
    if(number == 0 or number > 40):
        return '{:>2}'.format(element)
    elif(number == 2):
        return '{:>18}'.format(element)
    elif(number == 4 or number == 5 or number == 12 or number == 16):
        return '{:>6}'.format(element)
    #elif(number == 40):
    #    return '{:>3}'.format(element)
    elif(number > 29 and number < 32):
        return '{:>14}'.format(element)
    else:
        return '{:>11}'.format(element)

@jit(nopython=True)
def out_events(time_array1ns, time_array2ns):
    j = 0
    jj = 0
    iact_stereo_events_ids = []
    deltas = []

    for i in range(len(time_array1ns)):
        j = jj
        k = 0

        while (j < len(time_array2ns)):
            delta = time_array1ns[i] + 1410 - time_array2ns[j] - 2040
            if(delta < 1000 and delta > -1000):
                deltas.append(delta)
                if(k == 0):
                    jj = j
                iact_stereo_events_ids.append([i,j])
                k = k + 1
                if(time_array1ns[i] - time_array2ns[j+1] < 0):
                    break
            j = j + 1
    return iact_stereo_events_ids, deltas

def get_stereo_cr(idxs, time_array1ns, window):
    cr_time = time_array1ns[idxs[0][0]]
    cr_bank = 0
    cr_sh = 0
    cr_array = np.array([])
    for i,_ in idxs:
        if(time_array1ns[i] <= cr_time + window):
            cr_bank+=1
            if(i == idxs[-1][0]):
                cr_array = np.append(cr_array,np.full((1,cr_bank),round(float(cr_bank)/(float(window)/float(1e9)),2)))
                break
        else:
            print(cr_bank, cr_time,cr_time + window)
            cr_array = np.append(cr_array,np.full((1,cr_bank),round(float(cr_bank)/(float(window)/float(1e9)),2)))
            cr_time = cr_time + window
            cr_sh = cr_sh + cr_bank
            cr_bank = 1
            if(i == idxs[-1][0]):
                cr_array = np.append(cr_array, 0)
                break
    return cr_array

with open(sys.argv[1]) as file:
    paths1 = []
    paths2 = []
    outs = []
    plot = int(file.readline().split()[0])
    path1 = file.readline().split()[0]
    path2 = file.readline().split()[0]
    out_path = file.readline().split()[0]
    cleaning = file.readline().split()[0]
    file.readline()
    while file:
        header = file.readline()
        if not header:
            break
        print(path1 + header.split()[0] + '/' + header.split()[0] + '_out_hillas_' + cleaning + '.csv')
        print(path2 + header.split()[1] + '/' + header.split()[1] + '_out_hillas_' + cleaning + '.csv')
        print(out_path + header.split()[1][:-2] + header.split()[2])
        paths1.append(path1 + header.split()[0] + '/' + header.split()[0] + '_out_hillas_' + cleaning + '.csv')
        paths2.append(path2 + header.split()[1] + '/' + header.split()[1] + '_out_hillas_' + cleaning + '.csv')
        outs.append(out_path + header.split()[1][:-2] + header.split()[2])
        print(file)
if plot == 1:
    fig, ax = plt.subplots(figsize=(8, 8), dpi = 400)
print(paths1, paths2, outs)
log_file = open('./log/find_stereo_' + str(datetime.date(datetime.now())) + '_' + str(datetime.time(datetime.now())), 'w')
log_file.write('file1,file2 | number of stereo events\n')
for i in range(len(paths1)):
    hillas_line_iacts = [[],[]]
    time_array = [[],[]]
    for pii, pi in enumerate((paths1[i], paths2[i])):
        with open(pi) as file:
            header = file.readline()
            for header in file:
                if(len(header) > 0):
                    #time_array1s.append(math.modf(float(header.split(",")[2]))[1])
                    #print(pd.to_datetime(math.modf(float(header.split(",")[2]))[1], unit = 's') + pd.to_datetime(float(header.split(",")[3]), unit = 'ns'))
                    time_array[pii].append(np.int64(str(int(math.modf(float(header.split(",")[2]))[1])) + '{message:{fill}{align}{width}}'.format(message=int(float(header.split(",")[3])), fill='0', align='>', width=9)))
                    #print(np.int64(str(int(math.modf(float(header.split(",")[2]))[1])) + '{message:{fill}{align}{width}}'.format(message=int(float(header.split(",")[3])), fill='0', align='>', width=9)))
                    hillas_line_iacts[pii].append(header)
                    #t = Time(math.modf(float(header.split(",")[2]))[1] + float(header.split(",")[3])*1E-9, )
                    #print("{0:10d}.{1:9d}".format(int(math.modf(float(header.split(",")[2]))[1]), int(header.split(",")[3])))
                    #time = np.datetime64(np.int64(str(int(math.modf(float(header.split(",")[2]))[1])) + '{message:{fill}{align}{width}}'.format(message=int(float(header.split(",")[3])), fill='0', align='>', width=9)), 'ns')
                    #t = Time(time, format = "plot_date", scale='utc')
                    #t.format = 'iso'
                    #print(time)

    out_file = open(outs[i], 'w')

    iact_stereo_events_ids, deltas = out_events(time_array[0], time_array[1])
    if plot == 1:
        ax.set_title('Delays')
        ax.hist(deltas, range= [-1000, 1000], bins=200, label = paths1[i][-33:-27], histtype='step')
    if len(iact_stereo_events_ids) > 0:
        cr_array = get_stereo_cr(iact_stereo_events_ids, time_array[0], 120000000000.0)
        print(len(cr_array), len(iact_stereo_events_ids))
        for ii in range(len(iact_stereo_events_ids)):
            out = str(ii) + ',2\n'
            for j in range(2):
                out = out + str(j) + ',' + str(cr_array[ii]) +',' + hillas_line_iacts[j][iact_stereo_events_ids[ii][j]]
            out_file.write(out)
        print("--- %s seconds ---" % (time.time() - start_time))
    out_file.close()
    log_file.write(paths1[i][-33:-24] + ',' + paths2[i][-33:-24] + ' | ' + str(len(iact_stereo_events_ids)) + '\n')
if plot == 1:
    plt.legend(loc='upper left')
    ax.set(xlabel='(IACT01 - IACT02) delay, ns')
    ax.grid()
    plt.savefig('./delays_stereo.png')
