import sys
import os
import shutil

def FolderCopy(src, dist):
    for volume_folder in src:
        if (os.path.exists(volume_folder)):
            folder_name = volume_folder.split('\\')[-1]
            copy_path = dist + '\\' + folder_name
            try:
                shutil.copytree(volume_folder,copy_path)
                print('Finish' + 'copy_path')
            except:
                print('Failed to copy' + volume_folder)
        else:
            print('Failed to copy' + volume_folder)
    return True

def SelectOneVloume(path, search_keys, selected_path):
    for root, dirs, files in os.walk(path):
        for folder in dirs:
            sub_folder_full_path = os.path.join(root, folder)
            file_list = os.listdir(sub_folder_full_path)
            min_slice_number = int(search_keys[1])
            if(len(file_list) > min_slice_number):
                selected_path.append(sub_folder_full_path)
                return True
    return False

# Only select one volume from one patient
# search_keys
# 0 sequence name for example MPRAGE
# 1 slices number
# 2 search count
def SerachVolume(path, search_keys):
    log_path = path + '\\' + search_keys[0]+ '_volume_search_log.txt'
    file = open(log_path, 'w')
    selected_file_list = []
    search_count = 0;
    folder_list = os.listdir(path)
    volume_count = int(search_keys[2])
    sequence_name = search_keys[0]
    for folder in folder_list:
        folder_full_path =  path + '\\' + folder
        is_selected = False
        for root, dirs, files in os.walk(folder_full_path):
            if(is_selected):
                break
            for sub_folder in dirs:
                if(sub_folder == sequence_name):
                    specific_path = folder_full_path + '\\' + sub_folder
                    selected_path = []
                    if(SelectOneVloume(specific_path, search_keys, selected_path)):
                        selected_file_list.append(selected_path[0])
                        file.write(selected_path[0] + '\n')
                        search_count += 1
                        print(selected_path[0])
                        print(search_count)
                        is_selected = True
                        break
        if(search_count == volume_count):
           file.close()
           break          
    return selected_file_list 

search_keys =['MPRAGE','100', '50']
results = []
results = SerachVolume('\\\\storage.wsd.local\\Warehouse\\Data\\ADNI\\Normal\\ADNI', search_keys)
FolderCopy(results, '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\Normal')
results = []
results = SerachVolume('\\\\storage.wsd.local\\Warehouse\\Data\\ADNI_RENEW\\AD\\ADNI', search_keys)
FolderCopy(results, '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\AD')

print('Finish!')