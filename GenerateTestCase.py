import os
import datetime
import shutil

def GetParcels(parcel_info_root):
    parcels = []
    file = open(parcel_info_root,'r')
    while True:
        line = file.readline()
        if len(line) == 0:
            break;
        else:
            info = line.strip('\n')
            parcels.append(info.split('@')[0])
    file.close()
    return parcels

def CopyFiles(src,dist):
    for file in src:
        shutil.copy(file, dist)


def createParcelTestCase(nc_root, type, case_root, parcel_info_root):
    parcels = GetParcels(parcel_info_root)
    for parcel in parcels:
        file_list = []
        for root, dirs, files in os.walk(nc_root):
            for file in files:
                if file.split('_')[0] == parcel:
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)
                    print(file_path)
        num = len(file_list)
        train_num = int(num * 0.7)
        test_num = num - train_num
        train_list = file_list[0:train_num]
        test_list = file_list[train_num :num]
        train_root_dir = case_root + '\\' + 'TestCase_' + parcel + '\\' + 'Train' + '\\' + type 
        test_root_dir = case_root + '\\' + 'TestCase_' + parcel + '\\' + 'Test' + '\\' + type 
        if not os.path.isdir(train_root_dir):
            os.makedirs(train_root_dir)
        if not os.path.isdir(test_root_dir):
            os.makedirs(test_root_dir)
        CopyFiles(train_list, train_root_dir)        
        CopyFiles(test_list, test_root_dir)  

createParcelTestCase(r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\AD_MNI_Parcel_RAW','AD',
r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Test_Case_Parcel',
r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\log\parcel_range_result_100.txt')


createParcelTestCase(r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Normal_MNI_Parcel_RAW','NC',
r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Test_Case_Parcel',
r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\log\parcel_range_result_100.txt')

def createBrainTestCase(root, type, case_root):
    file_list = []
    for root, dirs, files in os.walk(root):
        for file in files:
                if file.split('.')[-1] == 'dcm':
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)
                    print(file_path)
    num = len(file_list)
    train_num = int(num * 0.7)
    test_num = num - train_num
    train_list = file_list[0:train_num]
    test_list = file_list[train_num :num]
    train_root_dir = case_root + '\\' + 'TestCase_' + 'Brain' + '\\' + 'Train' + '\\' + type 
    test_root_dir = case_root + '\\' + 'TestCase_' + 'Brain'+ '\\' + 'Test' + '\\' + type 
    if not os.path.isdir(train_root_dir):
         os.makedirs(train_root_dir)
    if not os.path.isdir(test_root_dir):
        os.makedirs(test_root_dir)
    CopyFiles(train_list, train_root_dir)        
    CopyFiles(test_list, test_root_dir)  


#createBrainTestCase(
#    r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\AD_MNI_DCM',
#    'AD',
#    r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case'
#  
#)
#createBrainTestCase(
#    r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case\Normal_MNI_DCM',
#    'NC',
#    r'\\storage.wsd.local\Warehouse\Data\zhujiangyuan\MNI_Test_150_Case'
#    
#)

print ('OK')