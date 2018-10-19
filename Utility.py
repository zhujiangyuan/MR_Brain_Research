
import os
import sys
import pydicom

def ReadInputFile(file_path, start_line = 1, end_line = None):
    output_pairs_list = [];
    openfile= open(file_path)
    file_buffer = openfile.readlines()
    openfile.close()
    row_number = len(file_buffer)

    if(start_line > row_number or start_line < 0):
        start_line = 0
    else :
        start_line -= 1
    if(end_line == None or end_line > row_number):
         end_line = row_number

    used_file_buffer = file_buffer[start_line:end_line]

    for line in used_file_buffer:
        line_string = line.strip('\n')
        pair = line_string.split('@@@@')
        output_pairs_list.append(pair)
    
    return output_pairs_list


def GetLabelFileName(loading_path):
    string_split_array = loading_path.split("\\")
    return string_split_array[-1]

def GetDataFolderName(loading_path):
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

def Remove(pairs_list, remove_segment_list, label_save_position, log_save_position, need_debug_info = True, need_log = True):
    if (need_log):
        logfile= open(log_save_position, 'a')
    file_count = 0
    for each_pair in pairs_list:
        if(need_debug_info):
            debug_info = '****Label_file_name_' + each_pair[0] + '@@@@' + 'Data_file_name_' + each_pair[1]
            print(debug_info)
        try:
            image_dcm = pydicom.dcmread(each_pair[1])
            label_dcm = pydicom.dcmread(each_pair[0])
            new_array = image_dcm.pixel_array.copy()
            label_array = label_dcm.pixel_array.copy()
        except:
            info = '***Error_Case_different_invalid_dicom'
            info += each_pair[0] 
            info += "@@@@"
            info += each_pair[1] + '\n' 
            if (need_debug_info):
                print(info)
            if (need_log):
                logfile.write(info)
            continue
        if(new_array.shape != label_array.shape):
            debug_info = '***Error_Case_different_image_size_';
            info = each_pair[0] + '_' + str(label_array.shape[0]) + '_' +str(label_array.shape[0]) 
            info += "@@@@"
            info += each_pair[1] + '_' + str(new_array.shape[0]) + '_' + str(new_array.shape[0]) + '\n' 
            if (need_debug_info):
                print(debug_info + info)
            if (need_log):
                logfile.write(info)
            continue
        for index, val in enumerate(label_dcm.pixel_array.flat):
            if(1 == remove_segment_list.count(val)):
                new_array.flat[index]=0 
        image_dcm.PixelData = new_array.tobytes() 
        label_full_name = GetLabelFileName(each_pair[0]) 
        suffix_index = label_full_name.rfind('.')
        new_dcm_file_name = label_full_name[0: suffix_index]
        new_dcm_file_name = new_dcm_file_name + "_" + "RemoveSkull" + '.dcm'
        new_dcm_folder = GetDataFolderName(each_pair[0])
        new_dcm_folder_path = new_dcm_folder.split('\\')[-1]
        new_dcm_folder_path = new_dcm_folder_path + "_" + "RemoveSkull"
        new_dcm_folder_path = label_save_position + "\\" + new_dcm_folder_path
        if(os.path.isdir(new_dcm_folder_path) == False):
            os.makedirs(new_dcm_folder_path)   
        save_path = new_dcm_folder_path + '\\' + new_dcm_file_name
        image_dcm.save_as(save_path) 
        file_count += 1;
        if(need_debug_info):
            debug_info = '***Finished files count:' + str(file_count)
            print(file_count)

search_result_save_position = '\\\\storage.wsd.local\\Warehouse\\Data\\AD_label_dat_path_pair.txt'
label_save_position = '\\\\storage.wsd.local\\Warehouse\\Data\\label_RemoveSkull'
log_save_position = '\\\\storage.wsd.local\\Warehouse\\Data\\log_AD_label_dat_path_pair.txt'

output_pairs_list = []
output_pairs_list = ReadInputFile(search_result_save_position,104228)
skull_value_array = [255,256,257]
Remove(output_pairs_list, skull_value_array, label_save_position, log_save_position)

def CheckFolder(ID, Checked_map):
    if(len(Checked_map) == 0):
        return True
    if(Checked_map.get(ID) == None):
        return True
    return False

def CheckVolumeData(input_folder_path):
    AD_ID_map = {}
    Noraml_ID_map = {}
    for root, dirs, files in os.walk(input_folder_path):
        for folder in dirs:
            sub_folder_full_path = (os.path.join(root, folder))
            full_path_name = sub_folder_full_path.split('_')
            type = full_path_name[1]
            ID = full_path_name[4] + '_' + full_path_name[5] + '_' + full_path_name[6]
            if(type == 'Normal'):
                if(CheckFolder(ID, AD_ID_map) == False):
                    print(sub_folder_full_path)                     
            if(type == 'RENNEW'):
                if(CheckFolder(ID, Noraml_ID_map) == False):
                    print(sub_folder_full_path)    

print("Finish!")

