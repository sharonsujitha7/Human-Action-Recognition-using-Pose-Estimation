import pandas as pd
convert = pd.read_csv("E:\\Pose_Estimation\\output\\04-20-10-32-54-644\\skeletons\\00212.txt/")
convert.to_csv('Convert.csv',index=None)