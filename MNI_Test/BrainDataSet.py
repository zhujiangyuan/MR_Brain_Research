import os
import torch
import torch.utils.data as data
import numpy as np
import pydicom
import ProgramException

class DataSet(data.Dataset):
    ''' '''
    min_sub_folder_number_2 = 2

    def __init__(
        self, 
        data_root_path, 
        data_folder_structure = [['NC', 'Normal'], ['AD', 'AD']],
    ):
         self.data_root_path = data_root_path
         self.data_folder_structure = data_folder_structure
         self.data, self.labels = self.getDataSet()
   
    def getDataSet(self):
        data = []
        labels = []
        sub_folder_num = len(self.data_folder_structure)
        if(sub_folder_num < DataSet.min_sub_folder_number_2):
            raise ProgramException('DataSet',
                'DataSet must have more than 1 sub data folder.\n Current sub data folder number is' + str(sub_folder_num) 
            )
        for  sub_folder_struct in self.data_folder_structure:
             if len(sub_folder_struct) != 2 :
                 raise ProgramException('DataSet','Invalid Sub folder structure') 
             sub_folder = sub_folder_struct[0]
             label = sub_folder_struct[1]
             sub_folder_full_path  = os.path.join(self.data_root_path, sub_folder)
             if not os.path.exists(sub_folder_full_path):
                 raise ProgramException('DataSet','The following sub folder is not exist:\n' + sub_folder_full_path)
             try:
               file_list = sorted(os.listdir(sub_folder_full_path))
             except:
                raise ProgramException('DataSet','The following folder has unexpected sub folder:\n' + sub_folder_full_path)
             input_number = len(file_list)
             if input_number <= 0:
                raise ProgramException('DataSet', 'The following sub folder is not exist:\n' + sub_folder_full_path)
             for file in file_list:
                file_full_path = os.path.join(sub_folder_full_path, file)
                if not os.path.exists(file_full_path):
                     raise ProgramException('DataSet', 'The following file is not exist:\n' + file_full_path)
                #if file.split('.')[0] == self.parcel_name :
                data.append(file_full_path)
                if label == 'AD':
                   labels.append(1)
                else:
                   labels.append(0)

        return data, labels

    def __getitem__(self, index):
        target = self.labels[index]
        image = self.getEachDcmData(index)
        image = torch.from_numpy(image.astype(np.float32))
        return image, target

    def __len__(self):
        return len(self.data)

    def getRawDataDimension(self):
        image, target = self.__getitem__(0)
        return image.shape

    def getEachDcmData(self, index):
        file_full_path = self.data[index];
        image_dcm = pydicom.dcmread(file_full_path)
        raw_buffer = image_dcm.pixel_array
        #rows, cols, slices = self.getRawDataDimension()
        #raw_buffer.reshape((rows, cols, slices))
        return raw_buffer
            
if __name__ == "__main__":   
    #Unit Test
    print('Start')   
    test_data = DataSet(r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\TestCase_Brain\Train')   
    test_data.__getitem__(0) 
