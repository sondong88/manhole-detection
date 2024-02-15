import sys
import os
import xml.etree.ElementTree as ET

import math
import numpy as np
import xlsxwriter
import pandas as pd

from geopy.geocoders import Nominatim
from geopy.geocoders import ArcGIS

from arcgis.gis import GIS
from arcgis.geocoding import reverse_geocode
from PyQt5.QtWidgets import (QMainWindow, QTextEdit,
    QAction, QFileDialog, QApplication)
def get_main_source_dir(root_dir=None, name = 'Open Directory'):
    if root_dir is None:
        root_dir = '/media'
    main_source_dir = (QFileDialog.getExistingDirectory(None,name, root_dir))
    return main_source_dir

def calculate_list_xy_point_newdata(filename_annotation, list_GPS, realwidth):
    annots = [os.path.join(filename_annotation, s) for s in os.listdir(filename_annotation)]
    points_xy = []
    annots.sort()
    for annot in annots:
        base = os.path.basename(annot)
        base = os.path.splitext(base)[0]
        base_list = base.split('_')
        distance = int(base_list[-1][1:6]) - 1

        et = ET.parse(annot)
        element = et.getroot()
        element_objs = element.findall('object')
        for element_obj in element_objs:
            element_bndbox = element_obj.find('bndbox')
            xmin = element_bndbox.find('xmin').text
            xmax = element_bndbox.find('xmax').text
            ymin = element_bndbox.find('ymin').text
            ymax = element_bndbox.find('ymax').text
            x_center = int((int(xmin) + int(xmax))*realwidth/4479/ 2.0) - realwidth/2.0
            y_center = 10000 - int((int(ymin) + int(ymax)) / 2.0)
            y_continue = distance*10000 + y_center

            y_milestone = list_GPS[np.argmin(np.abs(list_GPS[:,0] - y_continue)),0]
            y_center = y_continue - y_milestone
            points_xy.append([np.argmin(np.abs(list_GPS[:,0] - y_continue)), x_center * 0.001, y_center * 0.001, base])

    print('Calculating xy of manhole: Finished')
    return points_xy

def calculate_latitude_longitude(list_GPS_10m, points_xy):
    a = 6378137.0
    b = 6356752.3142
    points_xy_North = []
    for i, point_i_xy in enumerate(points_xy):
        x_north = point_i_xy[1] * math.cos(list_GPS_10m[point_i_xy[0], 4]) + point_i_xy[2] * math.sin(
            list_GPS_10m[point_i_xy[0], 4])
        y_north = point_i_xy[2] * math.cos(list_GPS_10m[point_i_xy[0], 4]) - point_i_xy[1] * math.sin(
            list_GPS_10m[point_i_xy[0], 4])

        X_north = (-math.sin(list_GPS_10m[point_i_xy[0], 2]) * x_north
                   - math.sin(list_GPS_10m[point_i_xy[0], 1]) * math.cos(list_GPS_10m[point_i_xy[0], 2]) * y_north) + \
                  list_GPS_10m[point_i_xy[0], 5]

        Y_north = (math.cos(list_GPS_10m[point_i_xy[0], 2]) * x_north
                   - math.sin(list_GPS_10m[point_i_xy[0], 1]) * math.sin(list_GPS_10m[point_i_xy[0], 2]) * y_north) + \
                  list_GPS_10m[point_i_xy[0], 6]

        Z_north = math.cos(list_GPS_10m[point_i_xy[0], 1]) * y_north + list_GPS_10m[point_i_xy[0], 7]

        longitude = math.atan(Y_north / X_north) * 180 / math.pi + 180

        r = math.sqrt(X_north ** 2 + Y_north ** 2)
        e_phay_2 = (a ** 2 - b ** 2) / (b ** 2)
        e_2 = (1 - (b ** 2 / (a ** 2)))
        F = 54 * (b ** 2) * (Z_north ** 2)
        G = r ** 2 + (1 - e_2) * (Z_north ** 2) - e_2 * (a ** 2 - b ** 2)
        c = (e_2 ** 2) * (F) * (r ** 2) / (G ** 3)
        s = (1 + c + (c ** 2 + 2 * c) ** (1 / 2)) ** (1 / 3)
        P = F / 3 / ((s + 1 + 1 / s) ** 2) / (G ** 2)
        Q = (1 + 2 * (e_2 ** 2) * P) ** (1 / 2)
        r0 = (-P * e_2 * r) / (1 + Q) + (
                    1 / 2 * (a ** 2) * (1 + 1 / Q) - P * (1 - e_2) * (Z_north ** 2) / Q / (1 + Q) - 1 / 2 * P * (
                        r ** 2)) ** (1 / 2)
        # U = ((r - e_2 * r0) ** 2 + (Z_north) ** 2) ** (1 / 2)
        V = ((r - e_2 * r0) ** 2 + (1 - e_2) * (Z_north) ** 2) ** (1 / 2)
        z0 = (b ** 2) * Z_north / a / V

        latitude = math.atan((Z_north + e_phay_2 * z0) / r) * 180 / math.pi

        points_xy_North.append([point_i_xy[3], round(latitude,10), round(longitude,10)])
    print('Calculating GPS of the manholes: Finished')
    return points_xy_North


def make_GPS_manhole_excel_file(file_path, submission_sum_list):
    workbook = xlsxwriter.Workbook(file_path)
    cell_format = workbook.add_format()
    cell_format.set_align('center')
    cell_format.set_align('vcenter')
    worksheet = workbook.add_worksheet()
    for i, row in enumerate(submission_sum_list):
        worksheet.write_row(i, 0 , row)
    workbook.close()
    print('Making excel file: Finished')


def get_map_image(latitude, longitude):
    df = pd.DataFrame(list(zip([latitude], [longitude], [3])),
               columns =['X', 'Y', 'size'])

    try:
        gis = ArcGIS(username='iris01708', password='iris010900!',  timeout=10, referer='http://www.example.com')
        location = gis.reverse([latitude, longitude], timeout=10)
        return location.raw['Match_addr']
    except:
        try:
            gis = GIS(username='iris01708', password='iris010900!')
            print(f"Connected to {gis.properties.portalHostname} as {gis.users.me.username}")
            location = reverse_geocode([longitude, latitude])
            return location['address']['Match_addr']
        except:
            print('using Nominatim(user_agent="myGeocode") for getting address')
            gis = Nominatim(user_agent="myGeocode")
            location = gis.reverse([latitude, longitude], timeout=10)
            address = (location.raw['address']['city'] + ' ' + location.raw['address']['district'] + ' ' + location.raw['address']['suburb'] + ' ' + location.raw['address']['road'])
            return address

def calculate_list_gps_IRIS_newGPS(GPS_path):
    pgs_frame = pd.read_csv(GPS_path, header=None, skiprows=[0,1])
    list_GPS = pgs_frame[[0,1,2,3]].to_numpy()
    list_heading = pgs_frame[[6]].to_numpy()

    list_GPS[:, 1:3] = np.radians(list_GPS[:, 1:3])
    list_heading = np.radians(list_heading)

    list_GPS_10m = list_GPS[:,1:]
    N_phi = []
    a = 6378137.0
    b = 6356752.3142
    for i, list_GPS_10m_i in enumerate(list_GPS_10m):
        N_phi_i = a/((1 - (1-(b**2)/(a**2)) * ((math.sin(list_GPS_10m_i[0]))**2))**(1/2))
    #     print(N_phi_i)
        N_phi.append(N_phi_i)

    N_phi = np.array(N_phi)
    np.reshape(N_phi, (-1,1))

    XYZ = np.zeros((N_phi.shape[0], 3))
    XYZ[:,0] = (N_phi + list_GPS_10m[:,2])*(np.cos(list_GPS_10m[:,0]))*(np.cos(list_GPS_10m[:,1]))
    XYZ[:,1] = (N_phi + list_GPS_10m[:,2])*(np.cos(list_GPS_10m[:,0]))*(np.sin(list_GPS_10m[:,1]))
    XYZ[:,2] = (N_phi*((b**2)/(a**2)) + list_GPS_10m[:,2])*(np.sin(list_GPS_10m[:,0]))

    list_GPS_full = np.concatenate((list_GPS, list_heading, XYZ), axis = 1)
    print(list_GPS_full)
    return list_GPS_full


def calculating_GPS_namhole_per_day(load_folder_path, load_annotations_folder_path, save_GPS_results_folder_path):
    folder_list = os.listdir(load_folder_path)

    for folder_list_i in folder_list:

        pgs_path = os.path.join(load_folder_path, folder_list_i, "GEOFOG3D/0/data/GEOFOG3D.csv")
        print(pgs_path)
        list_GPS_10m = calculate_list_gps_IRIS_newGPS(pgs_path)

        filename_annotations = os.path.join(load_annotations_folder_path, folder_list_i, 'annotations')
        list_xy_points = calculate_list_xy_point_newdata(filename_annotations, list_GPS_10m, realwidth=4479)

        list_GPS_results = calculate_latitude_longitude(list_GPS_10m, list_xy_points)

        save_manhole_map_folder = os.path.join(os.path.dirname(save_GPS_results_folder_path), 'GPS_result_map')

        for i, list_GPS_results_i in enumerate(list_GPS_results):
            [_, latitude, longitude] = list_GPS_results_i
            address = get_map_image(latitude, longitude)
            list_GPS_results[i].append(address)

        path_file = os.path.join(save_GPS_results_folder_path, '{}.xlsx'.format(folder_list_i))
        make_GPS_manhole_excel_file(path_file, list_GPS_results)
        print('saved:', os.path.join(save_GPS_results_folder_path, '{}.xlsx'.format(folder_list_i)))
            
        print('----------------------------------------------------------------------------------------------------')

def calculate_GPS(load_folder_path):
    # load_folder_path = get_main_source_dir(root_dir=None, name="Open manhole results Directory")
    s_date = os.path.basename(load_folder_path)
    s_survey = s_date + '_survey'
    # s_survey = s_date
    s_GPS = s_date + '_GPS'
    s_manhole_result = s_date + '_manhole_results'
    GPS_folder_path = os.path.join(load_folder_path, s_GPS)
    print("---------")

    save_GPS_results_folder_path = os.path.join(GPS_folder_path, '{}_GPS_results_address'.format(s_date))
    if not os.path.exists(save_GPS_results_folder_path):
        os.makedirs(save_GPS_results_folder_path)

    load_survey_folder_path = os.path.join(load_folder_path, s_survey)

    load_annotations_folder_path = os.path.join(load_folder_path, s_manhole_result)

    calculating_GPS_namhole_per_day(load_survey_folder_path, load_annotations_folder_path, save_GPS_results_folder_path)
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     main()
