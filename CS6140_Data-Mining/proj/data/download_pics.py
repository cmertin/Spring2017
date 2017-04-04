from __future__ import print_function, division
import pandas as pd
import urllib.request
import os

def IsFile(dir_name):

    if os.path.isdir(dir_name):
        return True
    else:
        cmd = "mkdir " + dir_name
        os.system(cmd)
        return True


location = ["Seattle", "Miami", "Tallahassee", "Raleigh", "Austin", "Los_Angeles", "Provo", "Denver", "Tampa", "SLC"]
ext = ".csv"
delim = '*'

for loc in location:
    csv_file = loc + ext
    print("Downloading from " + csv_file)
    counter = 0
    mo_df = pd.read_csv(open(csv_file,'rU'), sep=delim, encoding='utf-8', engine='c')
    mo_df.drop_duplicates()
    mo_df.dropna()
    if loc == "Provo":
        mo_df = mo_df.loc[mo_df["Mormon"] == True]
    else:
        mo_df = mo_df.loc[mo_df["Mormon"] == False]
    if IsFile(loc) is True:
        dir_loc = loc + '/'
        pics = mo_df["Pics_640"]
        for item in pics:
            sub = 0
            try:
                t = item.split(',')
            except:
                continue
            for t_ in t:
                temp = t_.replace('[', '')
                temp = temp.replace(']', '')
                temp = temp.strip()
                temp = temp.replace("\'", '')
                f_out = dir_loc + loc + str(counter) + '-' + str(sub) + ".jpg"
                try:
                    urllib.request.urlretrieve(temp, f_out)
                except:
                    pass
                sub += 1
            counter += 1
