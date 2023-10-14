import os
import sys
import pandas as pd

args = sys.argv

classnames = {
    '0-correct': '0',
    'fictivniye(fictitious)': '1',
    '1-not-on-the-break-stand': '1',
    '2-from-the-screen': '2',
    '3-from-the-screen+photoshop': '3',
    '4-photoshop': '4'}

if not len(args) == 2 :
    print('format: \"python create_csv.py dirname\".\nMake sure that the directories are named according to their class or subclass\nClass dirs must be parents of their subclasses\' dirs\nMAKE SURE THAT THERE ARE NO OTHER FILES IN THE DIR')
    sys.exit()


def read_files(directory) :
    labels = ['path', 'class', 'subclass']
    df = pd.DataFrame(columns=labels)


    for root, dirs, files in os.walk(directory) :
        for file in files :
            path = os.path.join(root, file)
            classname = os.path.basename(root) if os.path.basename(root) == '0' or os.path.basename(root) == '1' or os.path.basename(root) == '2' or os.path.basename(root) == '3' or os.path.basename(root) == '4' else classnames[os.path.basename(root)]
            subclass = '0'

            if not classname == '0' :                                 # if class is not '0', then change class to parent directory name, and subclass to this dirs name
                subclass = classname
                classname = os.path.dirname(root)
                classname = os.path.basename(classname) if os.path.basename(classname) == '1' or os.path.basename(classname) == '2' or os.path.basename(classname) == '3' or os.path.basename(classname) == '4' else classnames[os.path.basename(classname)]

            new_row = {'path': path, 'class': classname, 'subclass': subclass}
            df = pd.concat([df, pd.DataFrame([new_row], index=None)], ignore_index=True)
    
    return df
            


df = read_files(args[1])
df = df.reset_index()

print(df)

df.to_csv('data.csv', sep=';', index=False)