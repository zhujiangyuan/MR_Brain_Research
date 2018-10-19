import scipy.misc
import pydicom
import os
# Function:
# Search all DCM files under the 'input_folder_path', 
# Resize the image array based on 'resize_shape'
# Save PNG files under 'save_folder_path';Save PNG files under 'save_folder_resize_path'

# Note:
# Saved files keep original folder structure
# For Example
#  input_folder_path
#                   --A1
#                       ---File1.dcm
#  Save as:
#  save_folder_path
#                   --A1
#                       ---File1.png

def SaveDCM2PNG(input_folder_path, save_folder_path, save_folder_resize_path, resize_shape = []):
    save_file_count = 0;
    if(os.path.isdir(input_folder_path) == False):
        print("Invaild input folder path")
        return  False
    for root, dirs, files in os.walk(input_folder_path):
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
                    is_resize = False
                    new_array = image_dcm.pixel_array.copy()
                    if (len(resize_shape) == 2):
                        newshape = (resize_shape[0], resize_shape[1])
                        if(image_dcm.pixel_array.shape != newshape):
                            new_array = scipy.misc.imresize(image_dcm.pixel_array,newshape) 
                            is_resize =True
                except:
                    print("Failed to resize img")
                    print(file_path)
                    continue
                try:
                    folder_path = ''
                    if(is_resize == False):
                        folder_path = save_folder_path + '\\' + folder
                    else:
                        folder_path = save_folder_resize_path + '\\' + folder + '_' + str(image_dcm.pixel_array.shape[0]) + '_' + str(image_dcm.pixel_array.shape[1])
                    if(os.path.isdir(folder_path) == False):
                        os.makedirs(folder_path) 
                    save_file_path = folder_path + '\\' + file.split('.')[0] +'.png' 
                    scipy.misc.imsave(save_file_path,new_array)
                    save_file_count += 1;
                    print(save_file_path)
                    print(save_file_count)
                except:
                    print("Failed to save png")
                    print(file_path)
                    continue
    return True           

