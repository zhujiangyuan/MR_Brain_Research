import sys
import os
import pydicom

from SavePNG import SaveDCM2PNG 
source_postition = '\\\\storage.wsd.local\Warehouse\Data\zhujiangyuan\label_RemoveSkull'
save_postion = '\\\\storage.wsd.local\Warehouse\Data\zhujiangyuan\label_RemoveSkull_New_PNG'
save_resize_postion = '\\\\storage.wsd.local\Warehouse\Data\zhujiangyuan\label_RemoveSkull_Resize_PNG'

def CheckBufferSize(input_folder_path):
    shape_dict = {}
    for root, dirs, flies in os.walk(input_folder_path):
        for folder in dirs:
            sub_folder_full_path = (os.path.join(root, folder))
            file_list = os.listdir(sub_folder_full_path)
            for file in file_list:
                try:
                    file_path = (os.path.join(sub_folder_full_path, file))
                    image_dcm = pydicom.dcmread(file_path)
                except:
                    print("Failed to open DCM file")
                    print(file_path)
                    continue
                try:
                    map_size = len(shape_dict)
                    if(map_size != 0):  
                        if(shape_dict.get(image_dcm.pixel_array.shape) != None): 
                            number = shape_dict.get(image_dcm.pixel_array.shape)    
                            number += 1;
                            shape_dict[image_dcm.pixel_array.shape] = number
                        else:
                            shape_dict[image_dcm.pixel_array.shape] = 1
                    else:
                        shape_dict[image_dcm.pixel_array.shape] = 1
                except:
                    print("Failed to get dmc shape")
                    print(file_path)
                    continue
    print(shape_dict) 

#CheckBufferSize(source_postition)                    
#SaveDCM2PNG(source_postition, save_postion, save_resize_postion, [256,240])
for root, dirs, flies in os.walk('\\\\10.81.22.7\\sambashare\\DaJiang'):
        for folder in dirs:
            sub_folder_full_path = (os.path.join(root, folder))
            file_list = os.listdir(sub_folder_full_path)
            folder_name_array = folder.split('_')
            volume_id = ''
            if(len(folder_name_array) > 8):
               volume_id =  folder_name_array[-2]
               for file in file_list:
                    new_file_name = file
                    new_file_name = new_file_name.split('.')[0] + '_' +volume_id + '.png'
                    org_path = os.path.join(sub_folder_full_path, file)
                    new_path = os.path.join(sub_folder_full_path, new_file_name)
                    os.rename(org_path, new_path)


print('Finished')