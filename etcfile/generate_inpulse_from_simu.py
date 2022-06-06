from dae_parser import CameraPoints
import os, shutil

"""
This script create

scene folder
    |--------- scene.obj
    |--------- camera-keyframe.dae
    |--------- 1
            |--------- scene.png
            |--------- simu.sh
            |--------- hdr
            |--------- A0A1A2
                |--------- A0.npy
                |--------- A1.npy
                |--------- A2.npy
                |--------- GT_depth.npy
                |--------- phase_depth.npy
    |--------- 2
    |--------- 3
"""

objfile = '/home/saijo/labwork/simulator_origun/model/export_bathroom_small.obj'
daefile = '/home/saijo/labwork/simulator_origun/export_bathroom_small-camera_keyframe.dae'
savefolder = 'simulate_scene'
simuname = 'bathroom_small'

simupath = os.path.join(savefolder, simuname)
os.makedirs(simupath, exist_ok=True)
shutil.copyfile(objfile, os.path.join(simupath,os.path.basename(objfile)))
shutil.copyfile(daefile, os.path.join(simupath,os.path.basename(daefile)))
