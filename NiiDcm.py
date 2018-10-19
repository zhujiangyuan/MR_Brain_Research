import sys
import os
import pydicom
import dicom2nifti
import subprocess

def DcmSeries2NiiSingleFile(dcm_path, nii_path):    
    for root, dirs, files in os.walk(dcm_path):
        for sub_folder in dirs:
            sub_folder_full_path = os.path.join(root, sub_folder)
            file_list = os.listdir(sub_folder_full_path)
            if(len(file_list) > 50):
                dcm_series = [];
                for file in file_list:
                    if(file.split('.')[1] == 'dcm'):
                        try:
                           file_path = os.path.join(sub_folder_full_path, file)
                           single_dcm = pydicom.dcmread(file_path)
                        except:
                            print("Failed to open DCM file")
                            print(file_path)
                            continue
                        dcm_series.append(single_dcm)
                output_file = nii_path + '\\' + sub_folder + '.nii'
                try:
                    dicom2nifti.convert_dicom.dicom_array_to_nifti(dcm_series, output_file)
                except:
                    print("Failed to save NII file")
                    print(sub_folder_full_path)
                    continue

def RunCommandWithLog(command, print_msg=True):
     pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
     for line in iter(pipe.stdout.readline, b''):
        if print_msg:
            line = line.rstrip().decode('utf8')
            line.strip('\n')
            print(">>>", line)
     return True    

def Nii2Dcm(bat_path, nii_folder_path):
    file_list = os.listdir(nii_folder_path)
    for file in file_list:
        if(file.split('.')[-1] == 'nii'):
            command = bat_path + ' ' + nii_folder_path + '\\' + file + ' ' + '-mr'
            RunCommandWithLog(command)
            print(file)

#dcm_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\Normal'
#nii_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\Normal_NIFTI'               
#DcmSeries2NiiSingleFile(dcm_path, nii_path)
dcm_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\AD_new'
nii_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\AD_NIFTI'               
#DcmSeries2NiiSingleFile(dcm_path, nii_path)
Nii2Dcm('C:\\Users\\zhujiangyuan-PC\\bin\\mango-convert2dcm.bat', '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\AD_MNI_NIFTI')
Nii2Dcm('C:\\Users\\zhujiangyuan-PC\\bin\\mango-convert2dcm.bat', '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\Normal_MNI_NIFTI')


