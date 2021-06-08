# 시간 측정 자동화를 위한 데커레이터의 정의

from functools import wraps
import time

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1=time.time()
        result = fn(*args, **kwargs)
        t2=time.time()
        print(f"@timefn : {fn.__name__} took {t2-t1} seconds")
        return result
    return measure_time


x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -0.42193

@timefn
def calculate_z_serial_purepython(maxiter, zs, cs):
    output = [0] * len(zs)
    
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while (abs(z) < 2. and n < maxiter):

            z = z*z+c
            n += 1
        output[i]=n
    return output

def calc_pure_python(desired_width, max_iterations):
    x_step = (x2-x1)/desired_width
    y_step = (y2-y1)/desired_width
    x=[]
    y=[]
    
    ycoord=y1
    
    while ycoord < y2:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    
    #return x, y
    zs=[]
    cs=[]
    
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))
    print("Length of x : ", len(x))
    print("Total elements : ", len(zs))
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    secs=end_time-start_time
    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")
    
    
    #return output
    assert sum(output) == 33219980
    

if __name__=="__main__":
    calc_pure_python(1000, 300)
