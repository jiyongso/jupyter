import matplotlib 
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
 
# use LaTeX, choose nice some looking fonts and tweak some settings
matplotlib.rc('font', family='serif')
matplotlib.rc('font', size=16)
matplotlib.rc('legend', fontsize=16)
matplotlib.rc('legend', numpoints=1)
matplotlib.rc('legend', handlelength=1.5)
matplotlib.rc('legend', frameon=False)
matplotlib.rc('xtick.major', pad=7)
matplotlib.rc('xtick.minor', pad=7)
matplotlib.rc('text', usetex=True)
# matplotlib.rc('text.latex', 
#               preamble=[r'\usepackage[T1]{fontenc}',
#                         r'\usepackage{amsmath}',
#                         r'\usepackage{txfonts}',
#                         r'\usepackage{textcomp}'])

matplotlib.rc('figure', figsize=(12, 9))



# 상수 정의
deg_to_rad = np.pi/180.0
rad_to_deg = 180.0/np.pi

# R_out : external collimator 반지름 (mm)
# R_in : internal collimator 반지름 (mm)
# R_d : 검출기 반경 (mm) . 현재는 R_in/2.0 으로 설정.

R_out = 95.0
R_in = 25.0

R_d = 10.0
# detector pixel assignment
# Number of bins in x and y axis
Nbins = 100
y0 = np.linspace(-1.0*R_d, R_d, Nbins)
z0 = np.linspace(-1.0*R_d, R_d, Nbins)
y1, z1 = 0.5*(y0[1:]+y0[:-1]), 0.5*(z0[1:]+z0[:-1])
dy, dz = y0[1]-y0[0], z0[1]-z0[0]

YY, ZZ = np.meshgrid(y1, z1)
RR = np.sqrt(YY**2+ZZ**2)

# generate slit in degree

# slit_open_deg = 5.0
# slit_period_deg = 15.0
# slit_angle_start = -77.5

def generate_slit_test(slit_open_deg = 5.0, slit_period_deg=15.0, slit_angle_start=-77.5):
    
    slit_open_list_rad0 = []
    slit_open_list_deg0 = []
    for t_ang in np.arange(slit_angle_start, 90.0-slit_open_deg, slit_period_deg):
        slit_open_list_rad0.append((t_ang*deg_to_rad, (t_ang+slit_open_deg)*deg_to_rad))
        slit_open_list_deg0.append((t_ang, t_ang+slit_open_deg))
    return slit_open_list_deg0, slit_open_list_rad0

def generate_slit_kkh(N_slits=12, slit_deg=7.0, slat_deg=7.0):
    """
    김기현 교수 IEEE 2019 논문에 따른 slit angle 생성.
    
    N_slit : number of slit
    slit_deg : slit open angle (deg)
    slat_deg : slat angle (deg)
    """
    slit_cover = slit_deg*N_slits + slat_deg*(N_slits-1)
    slit_start = -slit_cover/2.0
    print(slit_cover, slit_start)
    return generate_slit_test(slit_deg, slit_deg+slat_deg, slit_start)
    
     
    
def get_closed_from_slit_open(open_deg):
    d0=-90.0
    d1=90.0
    result = []
    for sl in open_deg:
        result.append((d0, sl[0]))
        d0 = sl[1]
    result.append((d0, 90.0))
    return result
        
    
    

def plot_hemispherical_collimoator(R1, slits1, R2=None, slits2=None, view_ele=30.0, view_azim=45.0):
    th0 = np.linspace(0.0, np.pi, 100)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.view_init(view_ele, view_azim)
    cslits1 = get_closed_from_slit_open(slits1)
    cslits2 = get_closed_from_slit_open(slits2)
    for sl1 in cslits1:
        ph0 = np.linspace(sl1[0]*deg_to_rad, sl1[1]*deg_to_rad, 100)
        Theta, Phi = np.meshgrid(th0, ph0)
        X1 = R1 * np.sin(Theta) * np.cos(Phi)
        Y1 = R1 * np.sin(Theta) * np.sin(Phi)
        Z1 = R1 * np.cos(Theta)
        
        plot = ax.plot_surface(
            X1, Y1, Z1, rstride=1, cstride=1, color="b",
            linewidth=0, antialiased=False, alpha=0.3)
        
    for sl2 in cslits2:
        ph0 = np.linspace(sl2[0]*deg_to_rad, sl2[1]*deg_to_rad, 100)
        Theta, Phi = np.meshgrid(th0, ph0)
        X2 = R2 * np.sin(Theta) * np.cos(Phi)
        Y2 = R2 * np.sin(Theta) * np.sin(Phi)
        Z2 = R2 * np.cos(Theta)
        
        plot = ax.plot_surface(
            X2, Y2, Z2, rstride=1, cstride=1, color="r",
            linewidth=0, antialiased=False, alpha=0.4)
    
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
    

def get_solution2(a,b,c):
    """
    $ax^2+bx+c=0$ 의 2차방정식의 해를 return 한다.
    
    $D=\sqrt{b^2-4ac}$ 하 할 때,
    
    answer :
    
    (-b+D)/2, (-b-D)/2a
    
    """
    D=np.sqrt(b**2-4*a*c)
    
    return ((-b+D)/(2*a), (-b-D)/(2*a))


def cartesian_to_spherical2(x, y, z):
    """
    args
    ----
    x, y, z
    
    return sperical coordinate from given cartesian coordinate (x, y, z) 
    
    return
    ---
    (r, theta [deg], phi [deg])
    
    Warning : if x=y=z=0.0, return (0.0, 0.0, 0.0)
    """
    r=np.sqrt(x**2+y**2+z**2)
    theta = 0.0
    phi = 0.0
    if r==0.0:
        return (0.0, 0.0, 0.0)
    if z==0.0:
        theta = np.pi/2.0
    else :
        theta = np.arccos(z/r)
    if x == 0.0:
        phi = 0.0
    else :
        phi = np.arctan2(y, x)
    
    return (r, theta*rad_to_deg, phi*rad_to_deg)

def cartesian_to_spherical(x, y, z):
    """
    args
    ----
    x, y, z
    
    return sperical coordinate from given cartesian coordinate (x, y, z) 
    
    return
    ---
    (r, theta [deg], phi [deg])
    
    Warning : if x=y=z=0.0, return (0.0, 0.0, 0.0)
    """
    theta = np.zeros(x.shape)
    phi = np.zeros(x.shape)
    r=np.sqrt(x**2+y**2+z**2)
    r0ind=np.where(r==0.0)
    rind = np.where(r>0.0)
    x0ind = np.where(x==0.0)
    #xind = np.where(x!=0.0)
    theta[rind] = np.arccos(z[rind]/r[rind])
    phi= np.arctan2(y, x)
    
    return (r, theta*rad_to_deg, phi*rad_to_deg)


def spherical_to_cartesian(r, theta, phi):
    """
    args
    ----
    r, theta [deg], phi[deg]
    
    return sperical coordinate from given cartesian coordinate (x, y, z) 
    
    return
    ---
    (x, y, z)
    """
    th, ph = theta*deg_to_rad, phi*deg_to_rad
    
    return (r*np.sin(th)*np.cos(ph), r*np.sin(th)*np.sin(ph), r*np.cos(th))
    
    

# slit_open_list_rad = []
# slit_open_list_deg = []
# for t_ang in np.arange(slit_angle_start, 90.0-slit_open_deg, slit_period_deg):
#     slit_open_list_rad.append((t_ang*deg_to_rad, (t_ang+slit_open_deg)*deg_to_rad))
#     slit_open_list_deg.append((t_ang, t_ang+slit_open_deg))

slit_open_list_deg, slit_open_list_rad = generate_slit_kkh() 

def is_open(angle):
    """
    angle : angle in degree
    """
    result = False
    angrad = angle*deg_to_rad
    
    for angs in slit_open_list_rad:
        if angrad < angs[0]:
            pass
        elif (angrad >= angs[0]) and (angrad <= angs[1]):
            result = True
            break
        
    return result

# aa=np.arange(-90, 90.005, 0.01)
# ex_angle_mask =np.fromiter((is_open(ai) for ai in aa), aa.dtype)
# print(ex_angle_mask)


def generate_solid_angle(source_position):
    xs, ys, zs = source_position[0], source_position[1], source_position[2]
    n_area = dy * dz
    M_cos = xs/np.sqrt((xs)**2+(ys-YY)**2+(zs-ZZ)**2)
    
    return M_cos

def get_position_on_Rsphere_cartesian(source_position, Rsp, Y, Z):
    """
    hemisphere의 반경을 R 이라 하자. source가 (x, y, z)에 위치할 때 source에서 
    각 pixel의 중앙으로 직선을 그렸을 때 반경 R 인 구와 만나는 점의 3차원 좌표
    를 return 한다.
    
    args
    -------
    source_position : list/tuple/array with 3 elements in mm 
    Rsp : spherical collimator 반경
    Y : 각 pixel의 y값을 나타내는 nxn array
    Z : 각 pixel의 z값을 나타내는 nxn array
    
    return
    ------
    (XR, YR, ZR)
       
    """
    
    xs, ys, zs = source_position[0], source_position[1], source_position[2]
    
    A=xs**2+(ys-Y)**2+(zs-Z)**2
    B=2.0*(Y*(ys-Y)+Z*(zs-Z))
    #C=Y**2+Z**2-d_pixel**2
    C=Y**2+Z**2-Rsp**2
    
    t1, t2=get_solution2(A, B, C)
    
    XR=xs*t1
    YR=(ys-Y)*t1+Y
    ZR=(zs-Z)*t1+Z
    
    return (XR, YR, ZR)

def get_position_on_Rsphere_spherical(source_position, Rph, Y, Z):
    XR, YR, ZR = get_position_on_Rsphere_cartesian(source_position, Rph, Y, Z)
    #ex_angle_mask =np.fromiter((is_open(ai) for ai in aa), aa.dtype)
    M_R, M_theta, M_phi = cartesian_to_spherical(XR, YR, ZR)
    #print(M_R.max(), M_R.min(),M_theta.min(), M_theta.max(), M_phi.min(), M_phi.max())
    return M_theta, M_phi


def calc_valid_phi(source_position, d_pixel, Y, Z):
    Mth, Mph = get_position_on_Rsphere_spherical(source_position, d_pixel, Y, Z)
    M1=np.fromiter((is_open(mi) for mi in Mph.flatten()), Mph.dtype)
    return M1.reshape(Mph.shape)


def calc_rotation(sample_position = [65, 65, 0], rotation_step=1.0, rotation_direction=-1):
    zs, xs, ys = sample_position[0], sample_position[1], sample_position[2]
    rs = np.sqrt(xs**2+ys**2+zs**2)
    theta0=np.arctan2(np.sqrt(xs**2+ys**2), zs)
    phi0 = np.arctan2(ys, xs)
    newx=[]
    newy=[]
    newz=[]
    for dphi_deg in np.arange(0.0, 180.0+rotation_step/2.0, rotation_step):
        dphi = dphi_deg*deg_to_rad*(rotation_direction)
        nx = rs*np.cos(theta0)
        ny = rs*np.sin(theta0)*np.cos(phi0+dphi)
        nz = rs*np.sin(theta0)*np.sin(phi0+dphi)
        newx.append(nx)
        newy.append(ny)
        newz.append(nz)
    return np.array(newx), np.array(newy), np.array(newz)
        
        
        
def calc_main(sample_position = [65, 65, 0], rotation_step=1.0, rotation_direction=1, detector_size=10.0, Nbins=100):
    zs, xs, ys = sample_position[0], sample_position[1], sample_position[2]
    rs = np.sqrt(xs**2+ys**2+zs**2)
    theta0=np.arctan2(np.sqrt(xs**2+ys**2), zs)
    phi0 = np.arctan2(ys, xs)
    list_phi=[]
    list_count=[]
    
    
    y0 = np.linspace(-1.0*detector_size/2.0, detector_size/2.0, Nbins)
    z0 = np.linspace(-1.0*detector_size/2.0, detector_size/2.0, Nbins)
    y1, z1 = 0.5*(y0[1:]+y0[:-1]), 0.5*(z0[1:]+z0[:-1])
    dy, dz = y0[1]-y0[0], z0[1]-z0[0]

    Ym, Zm = np.meshgrid(y1, z1)
    
    for dphi_deg in np.arange(0.0, 360.0+rotation_step/2.0, rotation_step):
        dphi = dphi_deg*deg_to_rad*(rotation_direction)
        nx = rs*np.cos(theta0)
        ny = rs*np.sin(theta0)*np.cos(phi0+dphi)
        nz = rs*np.sin(theta0)*np.sin(phi0+dphi)
        source_position =[nx, ny, nz]
        M_theta0, M_phi0 = get_position_on_Rsphere_spherical(source_position , R_out, Ym, Zm)
        MK0 = calc_valid_phi(source_position, R_out, Ym, Zm)
        M_theta1, M_phi1 = get_position_on_Rsphere_spherical(source_position , R_in, Ym, Zm)
        MK1 = calc_valid_phi(source_position, R_in, Ym, Zm)
        MK=MK0*MK1
        list_phi.append(dphi)
        list_count.append(MK.sum())
    print("completed!!")
    return np.array(list_phi), np.array(list_count)