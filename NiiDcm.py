import sys
import os
import pydicom
import dicom2nifti
import subprocess
import gzip
import datetime 

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

def Dcm2Nii(bat_path, dcm_folder_path, nii_path, log_path = r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\log'):
    dcm_folder_list = os.listdir(dcm_folder_path)
    log_path = log_path + '\\' + 'Invalid_DCM_Volume_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
    file = open(log_path, 'w')
    for dcm_folder in dcm_folder_list:
        sub_folder_full_path = os.path.join(dcm_folder_path, dcm_folder)
        file_list = os.listdir(sub_folder_full_path)
        selected_file = ''
        for file in file_list:
            if(file.split('.')[-1] == 'dcm'):
               selected_file = file
               break
        file_full_path = os.path.join(sub_folder_full_path, selected_file)
        nii_file = dcm_folder + '.nii'
        output_full_path = os.path.join(nii_path, nii_file)
        if os.path.isfile(output_full_path) :
            continue
        dcm2nii_command = bat_path + ' ' + '-out' + ' ' + output_full_path + ' ' + file_full_path
        try:
            RunCommandWithLog(dcm2nii_command)
        except:
              print('Failed tp get nii:' + sub_folder_full_path)
              file.write(sub_folder_full_path + '\n')  
              continue 
        fz_full_path = output_full_path + '.gz'
        if os.path.isfile(fz_full_path):
            try:
                f_name = fz_full_path.replace(".gz", "")
                #获取文件的名称，去掉
                g_file = gzip.GzipFile(fz_full_path)
                #创建gzip对象
                open(f_name, "wb").write(g_file.read())
                #gzip对象用read()打开后，写入open()建立的文件中。
                g_file.close()
                os.remove(fz_full_path)
                print(fz_full_path)
            except:
                print('Failed tp get nii:' + sub_folder_full_path)
                file.write(sub_folder_full_path + '\n')  
                continue 
        else:
            file.write(sub_folder_full_path + '\n')
            print('Failed tp get nii gz:' + sub_folder_full_path)
    file.close()
        


         


#Dcm2Nii(
#    r'C:\Users\zhujiangyuan-PC\bin\mango-convert2nii.bat', 
#   r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Normal', 
#    r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Normal_NIFTI')
#Dcm2Nii(
#    r'C:\Users\zhujiangyuan-PC\bin\mango-convert2nii.bat', 
#    r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\AD', 
#   r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\AD_NIFTI')


#dcm_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\Normal'
#nii_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\Normal_NIFTI'               
#DcmSeries2NiiSingleFile(dcm_path, nii_path)
#dcm_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\AD_new'
##nii_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\AD_NIFTI'               
#DcmSeries2NiiSingleFile(dcm_path, nii_path)
Nii2Dcm('C:\\Users\\zhujiangyuan-PC\\bin\\mango-convert2dcm.bat', r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\AD_MNI_NIFTI')
Nii2Dcm('C:\\Users\\zhujiangyuan-PC\\bin\\mango-convert2dcm.bat', r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Normal_MNI_NIFTI')
print('Finish!')

