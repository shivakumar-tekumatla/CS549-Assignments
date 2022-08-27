import numpy as np
def half_disk(radius):
    a=np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    mask2 = x*x + y*y <= radius**2
    a[mask2] = 0
    b=np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    p = x>-1
    q = y>-radius-1
    mask3 = p*q
    b[mask3] = 0

    return p,q


# print(half_disk(2))

temp_array = np.full((5, 6), np.spacing(np.single(1)), dtype=np.float32)

print(temp_array)

