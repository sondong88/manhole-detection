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

def update_manual_checking(filename_program, dir_manhole_update_excel_folder, manhole_result_folder):
    print('---------------------------------')
    print('filename_program:', filename_program)
    excel_program = pd.read_excel(filename_program, header=None)
    excel_checking = pd.read_excel(os.path.join(manhole_result_folder, '{}_manhole_results_GPS/{}'.format(os.path.basename(manhole_result_folder), os.path.basename(filename_program))), header=None)
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
    for i in range(len(excel_checking)):
        excel_checking[1][i] = excel_program[1][i]
        excel_checking[2][i] = excel_program[2][i]
        excel_checking[9][i] = excel_program[9][i]

    excel_checking.to_excel(save_excel_path, index=None, header=None)

def update_manual_checking_list(folder_program_list, dir_manhole_update_excel_folder, manhole_result_folder):
    folder_programs = [os.path.join(folder_program_list, s) for s in os.listdir(folder_program_list)]
    folder_programs.sort()

    list_wrong = []
    for excel_files_program_i in folder_programs:
        try:
            update_manual_checking(excel_files_program_i, dir_manhole_update_excel_folder, manhole_result_folder)
        except:
            list_wrong.append(excel_files_program_i)

    return list_wrong


def main(args=None):
    dir_manhole_checking_folder = get_main_source_dir(root_dir=None, name="Open manual checking manhole Directory")
    manhole_result_folder = get_main_source_dir(root_dir=None, name="Open result manhole Directory")
    list_wrong_sum = []
    dir_manhole_checking_excel_folder = os.path.join(dir_manhole_checking_folder,
                                                     "{}_profile_excel".format(os.path.basename(dir_manhole_checking_folder)))
    dir_manhole_update_excel_folder = os.path.join(manhole_result_folder,
                                                   '{}_manhole_results_GPS_checking'.format(os.path.basename(manhole_result_folder)))
    if not os.path.exists(dir_manhole_update_excel_folder):
        os.makedirs(dir_manhole_update_excel_folder)
    list_wrong = update_manual_checking_list(dir_manhole_checking_excel_folder, dir_manhole_update_excel_folder, manhole_result_folder)
    list_wrong_sum.append(list_wrong)
    print('list_wrong:', list_wrong_sum)

#
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main()









