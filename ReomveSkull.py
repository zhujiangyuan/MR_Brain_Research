

import os
import sys
import pydicom

search_result_save_position = '\\\\storage.wsd.local\\Warehouse\\Data\\normal_label_dat_path_pair_Copy.txt'
label_save_position = '\\\\storage.wsd.local\\Warehouse\\Data\\label_RemoveSkull'
log_save_position = '\\\\storage.wsd.local\\Warehouse\\Data\\log_normal_label_dat_path_pair.txt'

openfile= open(search_result_save_position)
logfile= open(log_save_position, 'w')
label_data_pair_list =[]
while True:
    line = openfile.readline()
    if len(line) == 0:
        break
    else:
        line_string = line.strip('\n')
        pair = line_string.split('@@@@')
        label_data_pair_list.append(pair)
openfile.close()

def GetLabelFileName(loading_path):
    string_split_array = loading_path.split("\\")
    return string_split_array[-1]

def GetLabelFolderName1(loading_path):
    string_split_array = loading_path.split("\\")
    return string_split_array[-2]

def GetLabelFolderName(loading_path):
    string_split_array = loading_path.split("\\")
    folder_path = ''
    array_size = len(string_split_array)
    for index, val in enumerate(string_split_array.flat):
        if(index != array_size -1):
            folder_path += val
    return folder_path
file_count = 0;
for each_pair in label_data_pair_list:
    print(each_pair[0])
    print(each_pair[1])
    image_dcm = pydicom.dcmread(each_pair[1])
    label_dcm = pydicom.dcmread(each_pair[0])
    skull_value_array = [255,256,257]
    new_array = image_dcm.pixel_array.copy()
    label_array = label_dcm.pixel_array.copy()
    if(new_array.shape != label_array.shape):
        output_log = each_pair[0] + '_' + str(label_array.shape[0]) + '_' +str(label_array.shape[0]) 
        output_log += "@@@@"
        output_log += each_pair[1] + '_' + str(new_array.shape[0]) + '_' + str(new_array.shape[0]) + '\n' 
        logfile.write(output_log)
        continue
        
    for index1, val in enumerate(label_dcm.pixel_array.flat):
        if(1 == skull_value_array.count(val)):
           new_array.flat[index1]=0 
    image_dcm.PixelData = new_array.tobytes() 
    label_full_name = GetLabelFileName(each_pair[0]) 
    suffix_index = label_full_name.rfind('.')
    new_dcm_file_name = label_full_name[0: suffix_index]
    new_dcm_file_name = new_dcm_file_name + "_" + "RemoveSkull" + '.dcm'
    new_dcm_folder = GetLabelFolderName1(each_pair[0])
    new_dcm_folder_path = new_dcm_folder.split('\\')[-1]
    new_dcm_folder_path = new_dcm_folder_path + "_" + "RemoveSkull"
    new_dcm_folder_path = label_save_position + "\\" + new_dcm_folder_path
    if(os.path.isdir(new_dcm_folder_path) == False):
        os.makedirs(new_dcm_folder_path)   
    save_path = new_dcm_folder_path + '\\' + new_dcm_file_name
    image_dcm.save_as(save_path) 
    file_count += 1;
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(file_count)
logfile.close()
     







