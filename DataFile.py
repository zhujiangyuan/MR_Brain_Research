class DataFile:
    ''' This class represents Data file name '''
    slice_num_index_from_right = 3
    def __init__(self, full_name, folder_path, subject_ID, volume_key, sperator = '_', suffix = '.dcm'):
          self.full_name = full_name
          self.folder_path = folder_path
          self.suffix = suffix
          self.sperator = sperator
          self.subject_ID = subject_ID
          self.volume_key = volume_key
          self.loading_path = self.folder_path + '\\' + self.full_name
          self.parseFileName()

    def parseFileName(self):
        suffix_len = len(self.suffix)
        full_name_len = len(self.full_name)
        name_without_suffix = self.full_name[0:full_name_len - suffix_len]
        split_name_array = name_without_suffix.split(self.sperator)
        array_len = len(split_name_array)
        slice_num_pos = array_len - DataFile.slice_num_index_from_right
        self.slice_num = int(split_name_array[slice_num_pos])
        