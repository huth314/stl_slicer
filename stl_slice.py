import numpy as np
import pandas as pd
import scipy.optimize as so
from scipy.spatial import distance_matrix
from math import *
import sys

def show_bar(fraction, prefix='', width=60):
    x = int(fraction*width)
    print("{}[{}{}] {}/{}%".format(prefix, "#"*x, "."*(width-x), int(x/width*100), 100),
                end='\r', file=sys.stdout, flush=True)

class STLObject(object):
    '''! This class defines methods for slicing stl objects used for 3D FEBID.
    The slices can be used to generate a streamfile for Fisher Scientific SEMs with
    16-bit pattern generator.
    (c) Michael Huth, 2023.
    '''

    def __init__(self, stl_file='', scale=1.0):
        '''! Constructor of class.
        @param stl_file File name of stl-file to be read at instantiation. Can be empty.
        @param scale Scale factor to scale all triangles with.
        '''
        self.__scale_factor = scale # scale factor to be applied to triangle vertex coordinates.
        self.__stl_file = stl_file
        self.__triangles = [] # list of triangles to be read from stl-file.
        self.__normals = [] # list of normal vectors of triangle faces.
        self.__Lx, self.__Ly, self.__Lz = 0.0, 0.0, 0.0 # dimensions of stl object.
        if len(self.__stl_file) != 0:
            self.__setup_triangles()

    @property
    def stl_file(self):
        '''! Name of stl-file.
        @return String containing name of stl-file.
        '''
        return self.__stl_file

    @stl_file.setter
    def stl_file(self, stl_file, scale=1.0):
        '''! Setting stl-file after instatiation of stl-object.
        @param stl_file File name of stl-file to be read.
        @param scale Scale factor to scale all triangles with.
        '''
        self.__stl_file = stl_file
        self.__scale_factor = scale
        self.__setup_triangles()

    @property
    def scale_factor(self):
        '''! Scale factor.
        @return Float representing scale factor to be applied to coordinates of triangles.
        '''
        return self.__scale_factor

    @scale_factor.setter
    def scale_factor(self, scale):
        '''! Setting scale factor to be applied to coordinates of triangles.
        @param scale Scale factor.
        '''
        self.__scale_factor = scale
        if len(self.__stl_file) != 0:
            self.__setup_triangles()

    @property
    def triangles(self):
        '''! Triangles.
        @return List of triangles as read from stl-file.
        '''
        return self.__triangles

    @property
    def normals(self):
        '''! Normals of triangles.
        @return List of normal vectors of triangles.
        '''
        return self.__normals

    @property
    def width_x(self):
        '''! x-width.
        @return Width of stl object in x-direction.
        '''
        return self.__Lx

    @property
    def width_y(self):
        '''! y-width.
        @return Width of stl object in y-direction.
        '''
        return self.__Ly

    @property
    def width_z(self):
        '''! z-width.
        @return Width of stl object in z-direction.
        '''
        return self.__Lz

    def get_bounding_box(self):
        '''! Bounding box of stl object.
        @return Tuple of two numpy arrays representing (xmin, ymin, zmin) and (xmax, ymax, zmax).
        '''
        p1, p2 = self.__get_boundaries(self.__triangles)
        return (p1, p2)

    def __setup_triangles(self):
        '''Read triangles from stl-file and find their normal vectors.'''
        self.__read_triangles()
        self.__scale_triangles()
        self.__set_normals()

    def __read_triangles(self):
        '''! Read triangles from stl file.
        '''
        with open(self.__stl_file) as f:
            lines = f.readlines()
            inside = False
            triangles = []
            for line in lines:
                if line.find('outer loop') != -1:
                    inside = True
                    t = []
                elif line.find('endloop') != -1:
                    inside = False
                    triangles.append(t)
                elif line.find('vertex') != -1 and inside:
                    vxyz = line.strip('\n').split(' ', 4)
                    v = np.array([float(vxyz[1]), float(vxyz[2]), float(vxyz[3])])
                    t.append(v)
            p1, p2 = self.__get_boundaries(triangles)
            self.__triangles = []
            # shift triangle coordinates towards (0,0,0).
            for t in triangles:
                self.__triangles.append([t[0]-p1, t[1]-p1, t[2]-p1])
            self.__Lx, self.__Ly, self.__Lz  = (p2 - p1)[0], (p2 - p1)[1], (p2 - p1)[2]

    def __scale_triangles(self):
        tris = self.__triangles.copy()
        self.__triangles.clear()
        for t in tris:
            tt = []
            for v in t:
                tt.append(v*self.__scale_factor)
            self.__triangles.append(tt)
        self.__Lx *= self.__scale_factor
        self.__Ly *= self.__scale_factor
        self.__Lz *= self.__scale_factor

    def __set_normals(self):
        '''! Calculates normals for triangles.
        '''
        self.__normals = []
        for t in self.__triangles:
            v1, v2 = t[1] - t[0], t[2] - t[0]
            n = np.cross(v1, v2)/(sqrt(np.dot(v1, v1)*np.dot(v2, v2)))
            self.__normals.append(n)

    def __get_boundaries(self, triangles):
        '''! Determine dimensions in x, y and z.
        @param triangles List of triangles.
        '''
        x1 = x2 = triangles[0][0][0]
        y1 = y2 = triangles[0][0][1]
        z1 = z2 = triangles[0][0][2]
        for t in triangles:
            for p in t:
                x1, x2 = min(p[0], x1), max(p[0], x2)
                y1, y2 = min(p[1], y1), max(p[1], y2)
                z1, z2 = min(p[2], z1), max(p[2], z2)
        return (np.array([x1, y1, z1]), np.array([x2, y2, z2]))

    def single_slice(self, h, eps=1.0E-6):
        '''! Return list of points, lines and triangles as numpy arrays within cut at height z.
        Uses list of scaled triangles as stored in self.__scaled_triangles.
        @param h Height of cut; typically in units nm.
        @param eps Small parameter needed due to numerical rounding errors. Should be ok as pre-defined.
        @return List of points, lines and triangles as numpy arrays within cut plane at height z.
        '''
        if len(self.__triangles) == 0:
            return None
        elements = [] # will contain all vertices, lines and triangles in cut
        min_tilt = 1.0 # sine of triangle normal corresponding to most horizontal triangle cut by slice
        for k, t in enumerate(self.__triangles):
            v1, v2, v3 = t[0], t[1], t[2] # to reduce number of brackets in what follows
            h1, h2 = h - eps, h + eps
            if v1[2] < h1:
                state1 = 'b' # b: stands for below
            elif v1[2] > h2:
                state1 = 'a' # a: stands for above
            else:
                state1 = 'o' # o: stands for on
            if v2[2] < h1:
                state2 = 'b'
            elif v2[2] > h2:
                state2 = 'a'
            else:
                state2 = 'o'
            if v3[2] < h1:
                state3 = 'b'
            elif v3[2] > h2:
                state3 = 'a'
            else:
                state3 = 'o'
            if (state1 == 'a') and (state2 == 'a') and (state3 =='a'): # triangle above cut
                continue
            if (state1 == 'b') and (state2 == 'b') and (state3 == 'b'): # triangle below cut
                continue
            if (state1 == 'o') and (state2 == 'o') and (state3 == 'o'): # coplanar triangle on cut
                elements.append(('t', v1, v2, v3)) # t for triangle
                min_tilt = 0.0
                continue
            if (state1 == 'o') and (state2 == 'o'): # edge on cut
                elements.append(('l', v1, v2)) # l for line
                min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                continue
            if (state1 == 'o') and (state3 == 'o'): # edge on cut
                elements.append(('l', v1, v3))
                min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                continue
            if (state2 == 'o') and (state3 == 'o'): # edge on cut
                elements.append(('l', v2, v3))
                min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                continue
            if v1[2] != v2[2]:
                t12 = (h - v1[2])/(v2[2] - v1[2])
            else:
                t12 = -1
            if v1[2] != v3[2]:
                t13 = (h - v1[2])/(v3[2] - v1[2])
            else:
                t13 = -1
            if v2[2] != v3[2]:
                t23 = (h - v2[2])/(v3[2] - v2[2])
            else:
                t23 = -1
            if state1 == 'o': # v1 is on cut
                elements.append(('p', v1)) # p for point
                min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                if (t23 >= 0.0) and (t23 <= 1.0): # line opposite to v1 intersects cut
                    elements.append(('l', v1, v2 + t23*(v3 - v2)))
                    min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                    continue
            if state2 == 'o': # v2 is on cut
                elements.append(('p', v2))
                min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                if (t13 >= 0.0) and (t13 <= 1.0): # line opposite to v2 intersects cut
                    elements.append(('l', v2, v1 + t13*(v3 - v1)))
                    min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                    continue
            if state3 == 'o': # v3 is on cut
                elements.append(('p', v3))
                min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                if (t12 >= 0.0) and (t12 <= 1.0): # line opposite to v3 intersects cut
                    elements.append(('l', v3, v1 + t12*(v2 - v1)))
                    min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                    continue
            if state1 == state2: # v1 and v2 both above or below cut
                elements.append(('l', v1 + t13*(v3 - v1), v2 + t23*(v3 - v2)))
                min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                continue
            if state1 == state3: # v1 and v3 both above or below cut
                elements.append(('l', v1 + t12*(v2 - v1), v2 + t23*(v3 - v2)))
                min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                continue
            if state2 == state3: # v2 and v3 both above or below cut
                elements.append(('l', v1 + t12*(v2 - v1), v1 + t13*(v3 - v1)))
                min_tilt = min(min_tilt, sqrt(1.0 - self.__normals[k][2]**2))
                continue
        return (min_tilt, elements)

    def __bresenham(self, x1, y1, x2, y2):
        '''! Line between two points using Bresenham algorithm.
        @param x1 x-coordinate of endpoint 1.
        @param y1 y-coordinate of endpoint 1.
        @param x2 x-coordinate of endpoint 2.
        @param y2 y-coordinate of endpoint 2.
        @return List of 2-tuples representing raster points.
        '''
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx > dy:
            d1, d2 = dx, dy
            a, b, c, d = x1, y1, x2, y2
        else:
            d1, d2 = dy, dx
            a, b, c, d = y1, x1, y2, x2
        pk = 2*d2 - d1 # decision making parameter
        pts = []
        for i in range(0, d1 + 1):
            if dx > dy:
                pts.append((a,b))
            else:
                pts.append((b, a))
            if a < c:
                a += 1
            else:
                a -= 1
            if pk < 0:
                pk = pk + 2*d2
            else:
                if b < d:
                    b += 1
                else:
                    b -= 1
                pk = pk + 2*d2 - 2*d1
        return pts

    def __triangle_limits(self, v1, v2, v3):
        '''! Boundaries of triangle in xy-plane, i.e. coplananr triangle.
        @param v1 Numpy array for x- and y-coordinate of first corner point of triangle.
        @param v2 Second corner point of triangle.
        @param v3 Third corner point of triangle.
        @return 4-tuple for min, max in x- and y-direction, respectively.
        '''
        xmin = xmax = v1[0]
        ymin = ymax = v1[1]
        xmin = min(xmin, v2[0])
        xmin = min(xmin, v3[0])
        ymin = min(ymin, v2[1])
        ymin = min(ymin, v3[1])
        xmax = max(xmax, v2[0])
        xmax = max(xmax, v3[0])
        ymax = max(ymax, v2[1])
        ymax = max(ymax, v3[1])
        return (xmin, xmax, ymin, ymax)

    def __det2D(self, u, v):
        '''! 2D determinant of two-component columen vectors.
        @param u Numpy array representing two-component vector 1.
        @param v Numpy array representing two-component vector 2.
        @return Determinant of u and v.
        '''
        return u[0]*v[1] - u[1]*v[0]

    def __is_inside(self, v, v0, v1, v2):
        '''! Check whether 2D vertex is inside of triangle.
        @param v Numpy array representing vertex to be checked.
        @param v0 Numpy array representing first corner of triangle.
        @param v1 Second corner of triangle.
        @param v2 Third corner of triangle.
        @return True if v is inside triangle face (including edge).
        '''
        p1, p2 = v1 - v0, v2 - v0
        d12 = self.__det2D(p1, p2)
        d02 = self.__det2D(v0, p2)
        d01 = self.__det2D(v0, p1)
        dv2 = self.__det2D(v, p2)
        dv1 = self.__det2D(v, p1)
        a = (dv2 - d02)/d12
        b = -(dv1 - d01)/d12
        if (a >= 0) and (b>= 0) and (a + b <= 1.0):
            return True
        return False

    def raster_points_from_slice(self, h, pitch, eps=1.0E-6):
        '''! Return list of points as numpy arrays for slice at given height.
        Uses list scaled triangles as stored in self.__scaled_triangles.
        @param h Height of cut; typically in units nm.
        @param pitch x- and y step distance for raster grid; typically in unit nm.
        @param eps Small parameter needed due to numerical round errors. Should be ok as pre-defined.
        @return List of raster points as numpy arrays for x- and y-coordinates.
        '''
        min_tilt, elements = self.single_slice(h)
        pts = []
        for e in elements:
            if e[0] == 'p': # a point
                i, j = int(e[1][0]/pitch), int(e[1][1]/pitch)
                pts.append((i,j))
                continue
            elif e[0] == 'l': # a line
                x1, y1 = int(e[1][0]/pitch), int(e[1][1]/pitch)
                x2, y2 = int(e[2][0]/pitch), int(e[2][1]/pitch)
                lpts = self.__bresenham(x1, y1, x2, y2)
                for p in lpts:
                    pts.append(p)
            elif e[0] == 't': # a coplanar triangles
                x1, x2, y1, y2 = self.__triangle_limits(e[1], e[2], e[3])
                y = y1
                while y <= y2:
                    x = x1
                    while x <= x2:
                        v = np.array([x, y])
                        r = self.__is_inside(v, e[1], e[2], e[3])
                        if r == True:
                            i, j = int(v[0]/pitch), int(v[1]/pitch)
                            pts.append((i,j))
                        x += pitch
                    y += pitch
        u = pd.unique(pts)
        return (min_tilt, np.array([uu for uu in u]))

    def dist_matrix(self, pts, pitch, sigma):
        '''! Gauss-weighted 2D matrix of distance factors.
        @param pts List of points as numpy arrays from single slice.
        @param pitch Step distance in x- and y-direction (typically in nm).
        @param sigma FWHM of Gauss to be used for weighting (typically in nm).
        @return 2D matrix with weighting factors.
        '''
        d = distance_matrix(pts*pitch, pts*pitch, p=2) # this is fast!
        return np.exp(-d**2/(2*sigma**2))

    def dwell_time_factors(self, pts, pitch, dzp, sigma):
        '''! Dwell times for slice points in multiples of slice basis dwell time (typically 1 ms).
        @param pts List of ponts as numpy array from single slice.
        @param pitch Step distances in x- and y-direction, respectively (typically in nm).
        @param dzp Height increase per basis dwell time (typically in nm per ms).
        @param sigma FWHM of the Gaussian beam (typically in nm).
        @return List of dwell times for raster points in units of basis dwell time.
        '''
        m = self.dist_matrix(pts, pitch, sigma)*dzp
        tau, rnorm = so.nnls(m, np.ones(len(pts)))
        return tau

    def generate_streamfile(self, fname, HFW, pitch, dzp, dz_min, dz_max, sigma):
        '''! Writes stream file for the stl object.
        @param fname Name of streamfile to be generated.
        @param HFW Horizontal field width of SEM image (same units as pitch).
        @param pitch Pitch in x- and y-direction (typically in nm).
        @param dzp Height increase for slice (typically in nm).
        @param dz_min Mininum distance of adjacent slices in growth direction (same units as pitch).
        @param dz_max Maximum distance of adjacent slices in growth direction (same units as pitch).
        @param sigma FWHM of Gaussian beam (same units as pitch).
        '''
        if len(self.__triangles) == 0:
            return None
        f_handle = open(fname, 'w')
        f_handle.write('s16\n1\n') # for 16 bit pattern generator
        sf_content = [] # holds content to be added to streamfile
        p1, p2 = self.get_bounding_box()
        delta = dz_max - dz_min
        z = p1[2]
        print('Entering slicing loop ...')
        while z <= p2[2]:
            min_tilt, pts = self.raster_points_from_slice(z, pitch)
            tau = self.dwell_time_factors(pts, pitch, dzp, sigma)
            z += dz_min + min_tilt*delta
            self.__add_to_streamfile(sf_content, pts, tau, pitch, HFW)
            show_bar(z/p2[2], prefix='Writing slices')

        f_handle = open(fname, 'w')
        f_handle.write('s16\n1\n')
        n = '{:d}\n'.format(len(sf_content))
        f_handle.write(n)
        for s in sf_content:
            f_handle.write(s)
        f_handle.close()

    def __add_to_streamfile(self, sf, pts, tau, pitch, HFW):
        '''! Add entries for given points to streamfile list of strings.
        @param sf Streamfile name.
        @param pts List of points representing as 2-tuples (dimensionless).
        @param tau List of dwell times in units of basis dwell time (typically 1 ms).
        @param pitch Pitch distance in x- and y-direction (typically in nm).
        @param HFW Horizontal field width of SEM image (in same units as pitch).
        '''
        w2, h2 = self.width_x/2.0, self.width_y/2.0
        basis_factor = 10000
        for p, t in zip(pts, tau):
            if t*basis_factor < 1: # no dwell times below 100 ns
                continue
            x, y = int((0.5 + (p[0]*pitch - w2)/HFW)*65535), int((0.5 + (p[1]*pitch - h2)/HFW)*65535)
            s = '{:d} {:d} {:d}\n'.format(int(t*basis_factor), x, y)
            sf.append(s)
