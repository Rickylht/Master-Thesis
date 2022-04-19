import os
import csv

def get_path_list(path):
    list = []

    for root, ds, fs in os.walk(path):
        for f in fs:
            fullname = os.path.join(root, f)
            list.append(fullname)
    
    return list

def generate():
    path = ".\\teeth_dataset\\image"
    list = get_path_list(path)
    c = open(".\\teeth_dataset\\spreadsheet\\image.csv", "w", encoding='UTF8', newline='')
    writer=csv.writer(c)
    header = ['Id','ToothId','Wavelength','Gain','Moist','Polarization','Orientation']
    writer.writerow(header)
    for i in range(len(list)):
        _list = list[i].split('\\')
        id = i+1
        polarization = _list[3]
        wavelength = _list[4]
        gain = _list[5]
        tooth_id = _list[6]
        moist = _list[7]
        orientation = _list[8]
        #Id,ToothId,Wavelength,Gain,Moist,Polarization,Orientation
        row = [id,tooth_id,wavelength,gain,moist,polarization,orientation]
        writer.writerow(row)

    print("finished")

    
if __name__ == '__main__':
    generate()