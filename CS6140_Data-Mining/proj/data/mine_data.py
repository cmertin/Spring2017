import pynder
import urllib
import itertools
import csv
import os
from pathlib import Path
from time import sleep, gmtime, strftime
import subprocess


fb_email = "fb_email_login"
fb_pass = "fb_password_login"
FBID = "fb_acct_id"

cmd = "ruby tinder_fb_token.rb " + fb_email + ' ' + fb_pass
p = subprocess.Popen([cmd, ""], stdout=subprocess.PIPE, shell=True).communicate()
FBTOKEN = str(p[0].strip())[2:-1]
#print(FBTOKEN)
session = pynder.Session(facebook_id=FBID, facebook_token=FBTOKEN)

sleep_seconds = 5

lat = [47.6062, 25.761680, 30.438256, 35.779590, 30.267153, 34.082234, 40.233844, 39.761619, 27.950575, 40.758701] 
lon = [-122.3321, -80.191790, -84.280733, -78.638179, -97.743061, -118.243685, -111.658534, -104.962250, -82.457178, -111.876183] 
location = ["Seattle", "Miami", "Tallahassee", "Raleigh", "Austin", "Los_Angeles", "Provo", "Denver", "Tampa", "SLC"] 

for idx, loc in enumerate(location):

    out_file = loc + ".csv"
    out_file_ = Path(os.getcwd() + '/' + out_file)

    if out_file_.is_file() is False:
        with open(out_file, 'a') as outcsv:
            writer = csv.writer(outcsv, delimiter='*', lineterminator='\n')
            writer.writerow(["Name", "Age", "Schools", "Jobs", "Bio", "Mormon", "Pics_84", "Pics_172", "Pics_320", "Pics_640"])
    
    session.update_location(lat[idx], lon[idx])

    title = "\nLocation: " + loc + " - Output: " + out_file
    print(title)
    print('=' * len(title))

    total_users = 0
    
    while total_users < 50:
        user_data = []
        users = session.nearby_users(limit=25)
        total_users += len(users)
        t = strftime("    %Y-%m-%d %I:%M:%S %p") +  " - "
        t += "New Users: " + str(len(users)) + "\t Total Users: " + str(total_users)
        print(t)

        for user in itertools.islice(users, len(users)):
            mormon = False
            name = user.name
            age = user.age
            schools = user.schools
            jobs = user.jobs
            bio = user.bio
            pics_84 = user.get_photos(width="84")
            pics_172 = user.get_photos(width="172")
            pics_320 = user.get_photos(width="320")
            pics_640 = user.get_photos(width="640")
            if "lds" in bio.lower():
                if "not lds" not in bio.lower():
                    mormon = True
            for sc in schools:
                if "byu" in sc.lower():
                    mormon = True
                if "bringham" in sc.lower():
                    mormon = True
            temp = [name, age, schools, jobs, bio, mormon, pics_84, pics_172, pics_320, pics_640]
            user_data.append(temp)
            user.dislike()
            
        with open(out_file, 'a') as outcsv:
            writer = csv.writer(outcsv, delimiter='*', lineterminator='\n')
            for item in user_data:
                writer.writerow([item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9]])
                
        sleep(sleep_seconds)


