from scipy.spatial.transform import Rotation as R
import numpy as np

class RotVector:
    def __init__(self):
        self.lookat = np.array([0, 0, -1])
        self.lookup = np.array([0, 1, 0])

    def convert_blender2simu(self, blen_vec):
        simu_vec = np.array([blen_vec[0],blen_vec[2], -blen_vec[1]])
        return simu_vec

    def perspectiveLookatLookup(self, locvec, rotvec):
        r = R.from_rotvec(rotvec, degrees=True)
        lookat = self.convert_blender2simu(r.apply(self.lookat))
        lookup = self.convert_blender2simu(r.apply(self.lookup))

        return lookat, lookup

    def apply_coordinates_LookatLookup(self, locvec, rotvec):
        rx = R.from_rotvec(np.array([rotvec[0], 0, 0]).astype(float), degrees=True)
        ry = R.from_rotvec(np.array([0, rotvec[1], 0]).astype(float), degrees=True)
        rz = R.from_rotvec(np.array([0, 0, rotvec[2]]).astype(float), degrees=True)
        rot = rz * ry * rx
        lookat = self.convert_blender2simu(rot.apply(self.lookat)) + self.convert_blender2simu(locvec)
        lookup = self.convert_blender2simu(rot.apply(self.lookup)) + self.convert_blender2simu(locvec)

        return lookat, lookup

def orient_vec(origin, dist):
    v = dist - origin
    orinent = v / np.sqrt(np.sum(v ** 2))
    return orinent


if __name__ =='__main__':
    origin = np.array([0.7988916, 0.2083092, 1.765791])
    dist = np.array([-14.2, 11.1, -30])
    orient = orient_vec(origin, dist)
    print(orient)

    # print(37 * orient + origin)
    rotv = RotVector()
    rx = R.from_rotvec(np.array([107, 0, 0]).astype(float), degrees=True)
    ry = R.from_rotvec(np.array([0, 0, 0]).astype(float), degrees=True)
    rz = R.from_rotvec(np.array([0, 0, 25.2]).astype(float), degrees=True)
    rot = rz * ry * rx
    lookat = np.array([0, 0, -1])
    lookup = np.array([0, 1, 0])
    print(rotv.convert_blender2simu(rot.apply(lookat)))
    # lookat, lookup = rotv.apply_coordinates_LookatLookup(np.array([0.7988916, -1.765791, 0.2083092]).astype(float),
    #                                                      np.array([107, 0, 25.2]).astype(float))
    # print(lookat)

    # rotv = RotVector()
    # lookat, lookup = rotv.lookatLookup(45 * np.array([1, 0, 0]))
    #
    # print(lookat)
    # print(lookup)
    #
    # lookat, lookup = rotv.lookatLookup(45 * np.array([1, 0, 0]))
    #
    # print(lookat)
    # print(lookup)