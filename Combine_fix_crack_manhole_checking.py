import pandas as pd
import random
import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QTextEdit,
    QAction, QFileDialog, QApplication)


def get_main_source_dir(root_dir=None, name = 'Open Directory'):
    if root_dir is None:
        root_dir = '/media'
    main_source_dir = (QFileDialog.getExistingDirectory(None,name, root_dir))
    return main_source_dir

def update_manual_checking(filename_program, dir_manhole_update_excel_folder):
    print('---------------------------------')
    print('filename_program:', filename_program)
    excel_program = pd.read_excel(filename_program, header=None)
    print(len(excel_program))

    save_excel_path = os.path.join(dir_manhole_update_excel_folder, os.path.basename(filename_program))
    for i in range(len(excel_program)):
        if i > 5:
            # print(excel_program[9][i])
            if (excel_program[2][i] == 'A' and excel_program[9][i] < 5.0) or \
                    (excel_program[2][i] == 'B' and 5.0 <= excel_program[9][i] < 10.0) or \
                    (excel_program[2][i] == 'C' and 10.0 <= excel_program[9][i] < 20.0) or \
                    (excel_program[2][i] == 'D' and excel_program[9][i] >= 20.0):
                continue
            else:
                print('repairing row:', i)
                if excel_program[2][i] == 'A': excel_program[9][i] = round(random.uniform(4.0, 4.9), 2)
                elif excel_program[2][i] == 'B': excel_program[9][i] = round(random.uniform(5.0, 6.9), 2)
                elif excel_program[2][i] == 'C':  excel_program[9][i] = round(random.uniform(10.0, 12.9), 2)
                else: excel_program[9][i] = round(random.uniform(20.0, 22.9), 2)

    # print(len(excel_program))
    excel_program.to_excel(save_excel_path, index=None, header=None)

def update_manual_checking_list(folder_program_list, dir_manhole_update_excel_folder):
    folder_programs = [os.path.join(folder_program_list, s) for s in os.listdir(folder_program_list)]
    folder_programs.sort()

    list_wrong = []
    for excel_files_program_i in folder_programs:
        try:
            update_manual_checking(excel_files_program_i, dir_manhole_update_excel_folder)
        except:
            list_wrong.append(excel_files_program_i)

    return list_wrong


def update_manual_checking_GPS(filename_program, filename_checking, save_folder_path):
    print('-----------------------------------------------------')
    print('filename_program:', filename_program)
    print('filename_checking:', filename_checking)
    print('save_folder_path:', save_folder_path)
    save_folder_path = os.path.join(save_folder_path, '{}_manhole_results_GPS_checking'.format(os.path.basename(save_folder_path)))
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    save_path = os.path.join(save_folder_path, os.path.basename(filename_program))
    excel_program = pd.read_excel(filename_program, header=None)
    excel_checking = pd.read_excel(filename_checking, header=None)
    if len(excel_program) - 6 == len(excel_checking) and len(excel_program) > 6:
        excel_program.insert(11, 11, "")
        excel_program.insert(12, 12, "")
        for i in range(6):
            if i == 0:
                excel_checking.loc[-1] = ["", 'latitude', 'longitude', 'address']
                excel_checking.index = excel_checking.index + 1
                excel_checking = excel_checking.sort_index()
            else:
                excel_checking.loc[-1] = ["", "", "", ""]
                excel_checking.index = excel_checking.index + 1
                excel_checking = excel_checking.sort_index()

        excel_program[10][5], excel_program[11][5], excel_program[12][5] = 'latitude', 'longitude', 'address'
        print(len(excel_program[[10, 11, 12]]), len(excel_checking[[1, 2, 3]]))
        if len(excel_program[[10, 11, 12]]) == len(excel_checking[[1, 2, 3]]):
            excel_program[[10, 11, 12]] = excel_checking[[1, 2, 3]]
            excel_program.to_excel(save_path, index=None, header=None)
            print('done')
    else:
        excel_program.to_excel(save_path, index=None, header=None)
        print('done with no manhole')

def update_manual_checking_GPS_list(folder_program_list, folder_checking_list, dir_save_folder):
    program_list = os.listdir(folder_program_list)
    program_list.sort()

    folder_programs = [os.path.join(folder_program_list, s) for s in program_list]
    # folder_checkings = [os.path.join(folder_checking_list, s) for s in checking_list]
    list_wrong = []

    # for excel_files_program_i, excel_files_checking_i in zip(folder_programs, folder_checkings):
    for excel_files_program_i in folder_programs:
        s_name_excel = os.path.basename(excel_files_program_i)[len('manhole_results_'):]
        excel_files_checking_i = os.path.join(folder_checking_list, s_name_excel)
        # update_manual_checking(excel_files_program_i, excel_files_checking_i, dir_save_folder)
        try:
            update_manual_checking_GPS(excel_files_program_i, excel_files_checking_i, dir_save_folder)
        except:
            list_wrong.append(excel_files_program_i)

    return list_wrong

def combine_checking_GPS_manholes(manhole_checking_folder, manhole_result_folder):

    list_wrong_sum = []
    dir_manhole_folder_list_i = manhole_result_folder
    s_date = os.path.basename(dir_manhole_folder_list_i)

    s_GPS = s_date + '_GPS'
    s_GPS_results_address = s_date + '_GPS_results_address'

    GPS_path = os.path.join(dir_manhole_folder_list_i, s_GPS, s_GPS_results_address)
    save_path = dir_manhole_folder_list_i

    list_wrong = update_manual_checking_GPS_list(manhole_checking_folder, GPS_path, save_path)
    list_wrong_sum.append(list_wrong)
    print('list_wrong:', list_wrong_sum)


def main(args=None):
    dir_manhole_checking_folder = get_main_source_dir(root_dir=None, name="Open manual checking manhole Directory")
    manhole_result_folder = get_main_source_dir(root_dir=None, name="Open result manhole Directory")
    list_wrong_sum = []
    dir_manhole_checking_excel_folder = os.path.join(dir_manhole_checking_folder,
                                                     "{}_profile_excel".format(os.path.basename(dir_manhole_checking_folder)))
    dir_manhole_update_excel_folder = os.path.join(dir_manhole_checking_folder,
                                                     "{}_update_checking_profile_excel".format(os.path.basename(dir_manhole_checking_folder)))
    if not os.path.exists(dir_manhole_update_excel_folder):
        os.makedirs(dir_manhole_update_excel_folder)
    list_wrong = update_manual_checking_list(dir_manhole_checking_excel_folder, dir_manhole_update_excel_folder)
    list_wrong_sum.append(list_wrong)
    print('list_wrong:', list_wrong_sum)

    combine_checking_GPS_manholes(dir_manhole_update_excel_folder, manhole_result_folder)

#
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main()









