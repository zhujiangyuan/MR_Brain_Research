'''
Tools:
    Python-3.7
    Matlab SPM tool
    BGM BAT file
    Mango BAT file

Python library:
    pydicom
    Numpy
    
Basic Step:
    1. Select test data from ANDI database(DCM)
        (1) Search Keys
             Classification: AD, Normal...
             Sequence Type: MPRAGE...
             sample size： 50 or150...
        (2) Select only one volume from one patient
        (3) Log.Full path of each selected volume
    2. DCM -> NIFTI
       Each NIFTI single file reprsents a single volume.
       Using Mango BAT file
    3. MNI transformation
       Using Matlab SPM tools
    4. MNI_NIFTI->MNI_DCM
       Using Mango BAT file
    5. Run BGM
       Using BMG BAT files
    6. Extracl Parcel (Optional: only for parcel traning)
    7. GenerateTestCase
       Test_AnyParcel:
                     Train:
                            AD
                            Normal
                     Test:
Usage:
    Step(1)~Step(2),Step(4)~Step(7) could run automatically
    Step(3) only can run on  matlab PC.

Note:
    Each step still has somae special rules.Such as folder name , folder structure. 
            
'''