#-*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import pydicom
import os

def read_label(NC_label_dir, subject_name):#获取标签值

    '''
    根据病人编号读取label标签.
  
    Parameters:
      NC_label_dir - 字符串，label的存储路径
      subject_name - 字符串，病人编号
  
    Returns:
      each_data - numpy数组，label的3D数据，大小：156x189x157, 数值大小：0-289代表parcel编号。
  
    '''
    dcm_dir = os.path.join(NC_label_dir, subject_name)
    each_data = []
    for filename in os.listdir(dcm_dir):
        if '.dcm' not in filename:
            continue
        # 得到dicom的数据，每个切面大小：189x157
        path = os.path.join(dcm_dir, filename)
        dcm = pydicom.dcmread(path)
        ds = dcm.pixel_array  
        label_data = np.zeros(ds.shape, dtype=np.float32)
        label_data[:, :] = ds[:, :]
        each_data.append(label_data)
    each_data = np.array(each_data)
    return each_data

# 统计parcel所在位置
def static_parcel_location(NC_label_dir):
    
    '''
    根据label目录统计所有病人各个parcel的大小和范围
  
    Parameters:
      NC_label_dir - 字符串，label数据的存储路径
  
    Returns:
      parcel_size - 字典，存储各个parcel的大小（3个数值表示）
      parcel_interval - 字典，存储各个parcel的位置（6个数值表示）
  
    '''
    list_dir = sorted(os.listdir(NC_label_dir))
    parcel_dim_range={} #中间变量，保存每个样本各个parcel范围
    for i, subject_name in enumerate(list_dir):
        each_data = read_label(NC_label_dir, subject_name)#获取标签值
        
        for parcel_name in parcel_index.keys():
            parcel_dim_range[parcel_name] = []
            parcel_data = np.zeros_like(each_data)
            for index in parcel_index[parcel_name]:
                images = np.zeros_like(each_data)
                images[:,:,:] = each_data[:,:,:]
                images[images!=index] = 0
                images[images == index] = 1
                parcel_data = parcel_data + images
            d, h, w = np.nonzero(parcel_data)
            if len(d)==0 or len(h)==0 or len(w)==0:
                print(parcel_name)
            else:
                #前3个值：depth的大小，height的大小，width的大小
                #后6个值：d位置范围、h位置范围、w位置范围
                parcel_dim_range[parcel_name].append([np.max(d)-np.min(d)+1,
                                                     np.max(h)-np.min(h)+1,
                                                     np.max(w)-np.min(w)+1,
                                                     np.min(d), np.max(d),
                                                     np.min(h), np.max(h),
                                                     np.min(w), np.max(w)])
                                                     
    parcel_size = {}
    parcel_interval = {} #保存统计结果
    log_path = os.path.join(NC_label_dir, 'parcel_range.txt')
    fp = open(log_path, 'w')
    for parcel_name in parcel_index.keys():
        #save
        range_parcels = parcel_dim_range[parcel_name]
        info = '****'+parcel_name+':\n'
        fp.write(info)
        d_size = np.max(range_parcels, 0)[0]
        h_size = np.max(range_parcels, 0)[1]
        w_size = np.max(range_parcels, 0)[2]
        d_min = np.min(range_parcels, 0)[3]
        h_min = np.min(range_parcels, 0)[5]
        w_min = np.min(range_parcels, 0)[7]
        d_max = np.max(range_parcels, 0)[4]
        h_max = np.max(range_parcels, 0)[6]
        w_max = np.max(range_parcels, 0)[8]
        fp.write('    depth range:  ' + str(d_size)+'\n')
        fp.write('    height range:  ' + str(h_size)+'\n')
        fp.write('    width range:  ' + str(w_size)+'\n')
        fp.write('    depth interval: ' + str(d_min)
                                     +'-'+str(d_max)+'\n')
        fp.write('    height interval: ' + str(h_min)
                                      +'-'+str(h_max)+'\n')
        fp.write('    width interval: ' + str(w_min)
                                     +'-'+str(w_max)+'\n\n')
        parcel_size[parcel_name] = [d_size, h_size, w_size]
        parcel_interval[parcel_name] = [d_min, h_min, w_min, d_max, h_max, w_max]
    fp.close()
    
    return parcel_size, parcel_interval
    
    
#extract parcel
def extract_parcel(NC_raw_data_dir, parcel_interval, NC_parcel_dir, NC_label_dir):
    
    '''
    根据label目录统计所有病人各个parcel的大小和范围
  
    Parameters:
      NC_raw_data_dir - 字符串，配准后DICOM数据路径
      parcel_interval - 字典，存储各个parcel的位置（6个数值表示）
      NC_parcel_dir - 字符串，提取的parcel数据的存储路径
  
    Returns:
      None
      
    '''
    
    if not os.path.isdir(NC_parcel_dir):
            os.mkdir(NC_parcel_dir)
    NC_DCM_list = sorted(os.listdir(NC_raw_data_dir))
    for i, subject_name in enumerate(NC_DCM_list):
        print('NO.'+str(i)+' Save Patient '+subject_name[:-7]+'\'s parcel ...')
        filename = os.path.join(NC_raw_data_dir, subject_name)
        if '.dcm' not in filename:
            continue
        # 读取配准后的dicom的数据,存成3D，大小：156x189x157
        ds = pydicom.dcmread(filename).pixel_array    #读取原始数据
        # print(ds.shape, ds.min(), ds.max())
        label = read_label(NC_label_dir, subject_name[:-4])#获取标签值
        
        save_path = os.path.join(NC_parcel_dir, subject_name[:-4])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for parcel_name in parcel_index.keys():
            parcel_mask = np.zeros_like(label)
            for index in parcel_index[parcel_name]:
                tmp = np.zeros_like(label)
                tmp[:,:,:]=label[:,:,:]
                tmp[tmp!=index] = 0
                tmp[tmp==index] = 1
                parcel_mask = parcel_mask + tmp #计算parcel掩模
                
            parcel_data = parcel_mask*ds #掩模后的的图像
            #print(parcel_data, parcel_data.min(), parcel_data.max())
            
            p_range = parcel_interval[parcel_name]#parcel位置
            output_data = parcel_data[p_range[0]:p_range[3]+1,
                                      p_range[1]:p_range[4]+1,
                                      p_range[2]:p_range[5]+1]
            output_data.tofile(os.path.join(NC_parcel_dir, str(subject_name[:-4]),
                                            parcel_name+'Parcel.dat'))
    
if __name__ == "__main__":
    
    data_path = os.getcwd() #默认数据文件夹在当前路径
    #Normal
    #NC的label路径
    NC_label_dir = os.path.join(data_path, 'Normal_MNI_DCM_label_output')
    #NC的配准后数据的存储路径
    NC_raw_data_dir = os.path.join(data_path, 'Normal_MNI_DCM')
    #目标路径
    NC_parcel_dir = os.path.join(data_path, 'Normal_parcel')#保存地址
    
    ## AD
    #AD_label_dir = os.path.join(data_path, 'AD_MNI_DCM_label_output')
    #AD_raw_data_dir = os.path.join(data_path, 'AD_MNI_DCM')
    #AD_parcel_dir = os.path.join(data_path, 'AD_parcel')
        
    parcel_index={ #各个parcel对应的编号
        'Putamen': [79, 80],
        'Insular': [71, 72],
        'Cingulum': [139, 140, 141, 142],
        # 'Amygdala': [73, 74],
        # 'Thalamus': [83, 84],
        # 'LateralVentricle': [171,175,176, 177,178,179, 180, 181, 182, 183, 285, 286],
        'Hippocampus': [75, 76],
        # 'GlobusPallidu': [81, 82],
        # 'NucleusAccumbens': [89, 90],
        # 'CaudateNucleus': [77, 78],
        # 'EntorhinalArea': [47, 48]
    }
    ad_label_working_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\AD_MNI_DCM_label_output'
    ad_dcm_working_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\AD_MNI_DCM'
    ad_parcel_working_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\AD_MNI_DCM\\AD_MNI_Parcel'
   
    nc_label_working_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\Normal_MNI_DCM_label_output'
    nc_dcm_working_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\Normal_MNI_DCM'
    nc_parcel_working_path = '\\\\storage.wsd.local\\Warehouse\\Data\\zhujiangyuan\\MNI_Test_Case\\Normal_MNI_Parcel'
    #统计parcel结果
    parcel_size, parcel_interval = static_parcel_location(nc_label_working_path)
    #print(parcel_size, parcel_interval)
    
    #提取parcel并保存
    extract_parcel(nc_dcm_working_path, parcel_interval, nc_parcel_working_path, nc_label_working_path)
    
    