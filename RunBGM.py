import sys
import os
import subprocess

def RunCommandWithLog(command, print_msg=True):
     pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
     for line in iter(pipe.stdout.readline, b''):
        if print_msg:
            line = line.rstrip().decode('utf8')
            line.strip('\n')
            print(">>>", line)
     return True  

def RunBGM(BGM_path, dcm_folder_path):
    file_list = os.listdir(dcm_folder_path)
    for file in file_list:
        input = dcm_folder_path + '\\' + file
        file_name = file.split('.')[0]
        label_path = dcm_folder_path + '_label_output' + '\\' + file_name
        csv_path = label_path + '\\' + file_name + '.csv'
        if(os.path.isdir(label_path) == False):
           os.makedirs(label_path) 
        command = BGM_path + ' ' 
        command += '-input=' + input + ' '
        command += '-outputLabels=' + label_path + ' ' 
        command += '-outputCSV=' + csv_path + ' '
        command += '-iso=0' + ' ' 
        command += '-numAtlases=7' + ' '
        command += '-clear=0'
        RunCommandWithLog(command)

#RunBGM('\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\BGM\\TmvsBGM.exe', '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\Normal_MNI_DCM')
RunBGM('\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\BGM\\TmvsBGM.exe', r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Normal_MNI_DCM')
RunBGM('\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\BGM\\TmvsBGM.exe', r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\AD_MNI_DCM')