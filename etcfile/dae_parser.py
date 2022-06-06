from bs4 import BeautifulSoup
from create_lookat_lookup import RotVector
import numpy as np

class CameraPoints:
    def __init__(self, dae_file):
        #points_num is how many points does this data have.
        self.anim_num = 0 #total
        self.x_loc = []
        self.y_loc = []
        self.z_loc = []

        self.x_rot = []
        self.y_rot = []
        self.z_rot = []

        with open(dae_file, "r") as file:
            animetions = BeautifulSoup(file, "lxml")

        anim_num = animetions.find(id='Camera_CameraAction_location_X-output-array')
        num = int(anim_num.get('count'))
        self.anim_num += num

        x_loc_elem = animetions.find(id='Camera_CameraAction_location_X-output-array')
        if x_loc_elem:
            self.x_loc.extend(x_loc_elem.text.split())
        else:
            self.x_loc.extend([0 for i in range(num)])

        y_loc_elem = animetions.find(id='Camera_CameraAction_location_Y-output-array')
        if y_loc_elem:
            self.y_loc.extend(y_loc_elem.text.split())
        else:
            self.y_loc.extend([0 for i in range(num)])

        z_loc_elem = animetions.find(id='Camera_CameraAction_location_Z-output-array')
        if z_loc_elem:
            self.z_loc.extend(z_loc_elem.text.split())
        else:
            self.z_loc.extend([0 for i in range(num)])

        x_rot_elem = animetions.find(id='Camera_CameraAction_rotation_euler_X-output')
        if x_rot_elem:
            self.x_rot.extend(x_rot_elem.text.split())
        else:
            self.x_rot.extend([0 for i in range(num)])

        y_rot_elem = animetions.find(id='Camera_CameraAction_rotation_euler_Y-output')
        if y_rot_elem:
            self.y_rot.extend(y_rot_elem.text.split())
        else:
            self.y_rot.extend([0 for i in range(num)])

        z_rot_elem = animetions.find(id='Camera_CameraAction_rotation_euler_Z-output')
        if z_rot_elem:
            self.z_rot.extend(z_rot_elem.text.split())
        else:
            self.z_rot.extend([0 for i in range(num)])
            self.z_rot.extend([0 for i in range(num)])

    def show(self):
        rotv = RotVector()

        for xl, yl, zl, xr, yr, zr in zip(self.x_loc, self.y_loc, self.z_loc, self.x_rot, self.y_rot, self.z_rot):
            print(rotv.convert_blender2simu(np.array([xl, yl, zl]).astype(float)))
            lookat, lookup = rotv.apply_coordinates_LookatLookup(np.array([xl, yl, zl]).astype(float),
                                                         np.array([xr, yr, zr]).astype(float))
            print(lookat, lookup)

    def cam_loc_lookat_lookup_generator(self):
        rotv = RotVector()

        for xl, yl, zl, xr, yr, zr in zip(self.x_loc, self.y_loc, self.z_loc, self.x_rot, self.y_rot, self.z_rot):
            cam_loc = rotv.convert_blender2simu(np.array([xl, yl, zl]).astype(float))
            lookat, lookup = rotv.apply_coordinates_LookatLookup(np.array([xl, yl, zl]).astype(float),
                                                         np.array([xr, yr, zr]).astype(float))
            yield cam_loc, lookat, lookup




cam = CameraPoints(r"/home/saijo/labwork/研究結果まとめ/bathroom-keyframes-lookup.dae")
# cam.append_points(r"/home/saijo/labwork/研究結果まとめ/bathroom-keyframes-lookup.dae")
for cam_loc, lookat, lookup in cam.cam_loc_lookat_lookup_generator():
    print(cam_loc, lookat, lookup)
# cam.show()