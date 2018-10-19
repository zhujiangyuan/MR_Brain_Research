import sys
import os
import pydicom
from LabelFile import LabelFile
from DataFile import DataFile

label_source_position = '\\\\storage.wsd.local\\Warehouse\\Data\\label'
data_source_position = "\\\\storage.wsd.local\Warehouse\Data\ADNI_RENEW"
search_result_save_position = '\\\\storage.wsd.local\\Warehouse\\Data\\AD_label_dat_path_pair.txt'


label_sub_folder_list = os.listdir(label_source_position)

def GetDataFileFullPath(label_volume_folder):
    suffix_index = label_volume_folder.rfind('.')
    suffix = label_volume_folder[suffix_index + 1: len(label_volume_folder)]
    label_file_name = label_volume_folder[0: suffix_index]
    split_name_array = label_file_name.split('_')
    subject_ID = split_name_array[3] + '_' + split_name_array[4] + '_' + split_name_array[5]
    return subject_ID, suffix.split('_')[1]

volume_map = {}

for each_subject in label_sub_folder_list:
    (subject_ID, volume_key) = GetDataFileFullPath(each_subject)
    subject_path = label_source_position + '\\' + each_subject
    label_file_list = os.listdir(subject_path)
    labelFile_array =[]
    for each_lable in label_file_list:
        labelFile_array.append(LabelFile(each_lable, subject_path, subject_ID, volume_key))
    map_size = len(volume_map)
    if(map_size != 0):
       temp = volume_map.get(each_subject) 
       if(temp == None):
          volume_map[each_subject] = labelFile_array
       else:
          volume_map.get(each_subject).append(labelFile_array)
    else:
       volume_map[each_subject] = labelFile_array 

file = open(search_result_save_position, 'w')
pair_list = []
file_count = 0
handled_count = 0
for root, dirs, files in os.walk(data_source_position):
    for sub_folder in dirs:
        print(sub_folder)
        sub_folder_full_path = (os.path.join(root, sub_folder))
        sub_folder_path_array = sub_folder_full_path.split('\\')
        #\\storage.wsd.local\Warehouse\Data\ADNI\Normal\ADNI\002_S_0295\MPRAGE\2011-06-02_07_58_50.0\S110476
        sub_folder_key = ''
        array_size = len(sub_folder_path_array)
        if(array_size > 10):
            for inx, val in enumerate(sub_folder_path_array):
                if(inx >= 5):
                    sub_folder_key+=val
                if(inx >= 5 and inx != array_size -1):
                    sub_folder_key+= '_'
            if(volume_map.get(sub_folder_key) != None):
                handled_count = handled_count + 1
                print(sub_folder_key + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print(handled_count)
                label_object_array = volume_map.get(sub_folder_key)
                sub_folder_full_path = (os.path.join(root, sub_folder))
                data_file_list = os.listdir(sub_folder_full_path)
                for each_file in data_file_list:
                    print(each_file)
                    suffix_index = each_file.rfind('.')
                    suffix = each_file[suffix_index + 1: len(each_file)]
                    file_name = each_file[0: suffix_index]
                    if(suffix != 'dcm'):
                        continue
                    split_name_array = file_name.split('_')
                    data_slice_index = int(split_name_array[-3]) - 1
                    for each_label_object in label_object_array:
                        if(each_label_object.slice_num == data_slice_index):
                            data_object = DataFile(each_file, sub_folder_full_path, 0, volume_key);
                            pair_list.append([each_label_object.loading_path, data_object.loading_path])
                            file.write(each_label_object.loading_path + '@@@@' + data_object.loading_path + '\n')
                            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                            file_count = file_count + 1
                            print(file_count)
                            print(each_label_object.loading_path + '@@@@' + data_object.loading_path)
file.close()    
print('Finished')