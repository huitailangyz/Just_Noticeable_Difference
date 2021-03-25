from PIL import Image
import numpy as np
import skimage
from skimage import feature

eps = 1e-6

def conv(input, filter, padding="SAME", rotate=False):
    h, w = input.shape
    f_h, f_w = filter.shape
    if padding == 'VALID':
        pad_h, pad_w = 0, 0 
    elif padding == 'FULL':
        pad_h, pad_w = f_h-1, f_w-1
    else:
        pad_h, pad_w = f_h // 2, f_w // 2
    input_temp = np.zeros((h+2*pad_h, w+2*pad_w))
    input_temp[pad_h:pad_h+h, pad_w:pad_w+w] = input
    output = np.zeros((h-f_h+1+2*pad_h, w-f_w+1+2*pad_w))
    if rotate:
        filter = filter[::-1, ::-1]

    for i in range(f_h//2, h-f_h//2+2*pad_h):
        for j in range(f_w//2, w-f_w//2+2*pad_w):
            # print(i, j)
            output[i-f_h//2, j-f_w//2] = np.sum(input_temp[i-f_h//2: i+f_h//2+1, j-f_w//2: j+f_w//2+1] * filter)
    return output
    
#######################################################

def func_bg_adjust(bg_lum, min_lum):
    # adjust the luminance on dark region~(near 0)
    adapt_bg = np.round(min_lum + bg_lum*(127-min_lum)/127 + eps)
    bg_lum = np.where(bg_lum <= 127, adapt_bg, bg_lum)
    return bg_lum

def lum_jnd():
    # visuable threshold due to bg lum.
    bg_jnd = {} 
    T0 = 17
    gamma = 3 / 128
    for k in range(256):
        if k < 127:
            bg_jnd[k] = T0 * (1 - np.sqrt(k/127)) + 3 # NOTE: different from the paper
        else:
            bg_jnd[k] = gamma * (k-127) + 3
    return bg_jnd

def func_bg_lum_jnd(img):
    # Equ. 13
    min_lum = 32
    alpha = 0.7
    B = np.array([[1, 1, 1, 1, 1],
                  [1, 2, 2, 2, 1],
                  [1, 2, 0, 2, 1],
                  [1, 2, 2, 2, 1],
                  [1, 1, 1, 1, 1]])
    bg_lum = np.floor(conv(img, B) / 32)
    bg_lum = func_bg_adjust(bg_lum, min_lum)
    lum_jnd_table = lum_jnd()
    jnd_lum = np.vectorize(lum_jnd_table.get)(bg_lum)
    jnd_lum_adapt = alpha * jnd_lum
    return jnd_lum_adapt

#################################################
def gkern(kernlen=21, nsig=3):
    # Returns a 2D Gaussian kernel array.
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def func_edge_height(img):
    G1 = np.array([[0, 0, 0, 0, 0],
                   [1, 3, 8, 3, 1],
                   [0, 0, 0, 0, 0],
                   [-1, -3, -8, -3, -1],
                   [0, 0, 0, 0, 0]])
    G2 = np.array([[0, 0, 1, 0, 0],
                   [0, 8, 3, 0, 0],
                   [1, 3, 0, -3, -1],
                   [0, 0, -3, -8, 0],
                   [0, 0, -1, 0, 0]])
    G3 = np.array([[0, 0, 1, 0, 0],
                   [0, 0, 3, 8, 0],
                    [-1, -3, 0, 3, 1],
                    [0, -8, -3, 0, 0],
                    [0, 0, -1, 0, 0]])
    G4 = np.array([[0, 1, 0, -1, 0],
                   [0, 3, 0, -3, 0],
                   [0, 8, 0, -8, 0],
                   [0, 3, 0, -3, 0],
                   [0, 1, 0, -1, 0]])
    # calculate the max grad
    grad=np.zeros((*img.shape,4))
    grad[:,:,0] = conv(img, G1) / 16
    grad[:,:,1] = conv(img, G2) / 16
    grad[:,:,2] = conv(img, G3) / 16
    grad[:,:,3] = conv(img, G4) / 16
    max_gard = np.max(np.abs(grad), axis=2)
    maxgard = max_gard[2:-2, 2:-2]
    edge_height = np.pad(maxgard, ((2,2), (2, 2)), 'symmetric')
    return edge_height


def func_edge_protect(img):
    edge_h = 60
    edge_height = func_edge_height(img)
    max_val = np.max(edge_height)
    edge_threshold = edge_h / max_val
    if edge_threshold > 0.8:
        edge_threshold = 0.8
    edge_region = feature.canny(img, sigma=np.sqrt(2), low_threshold=0.4*edge_threshold*255, high_threshold=edge_threshold*255).astype(np.float32) # NOTE: result little different from matlab
    kernel = skimage.morphology.disk(3)
    img_edge = skimage.morphology.dilation(edge_region, kernel)
    img_supedge = 1-1*np.double(img_edge)
    gaussian_kernal = gkern(5, 0.8)
    edge_protect = conv(img_supedge, gaussian_kernal)
    return edge_protect

############################################################

def func_luminance_contrast(img):
    # calculate the luminance contrast for each pixel (Equ. 7)
    R = 2
    ker = np.ones((2*R+1, 2*R+1)) / (2*R+1)**2
    mean_mask = conv(img, ker) # mean value of each pixel
    mean_img_sqr = mean_mask**2 # square mean
    img_sqr = img**2 # square
    mean_sqr_img = conv(img_sqr, ker) # mean square
    var_mask = mean_sqr_img - mean_img_sqr # variance
    var_mask[var_mask<0] = 0
    valid_mask = np.zeros_like(img)
    valid_mask[R:-R,R:-R] = 1
    var_mask = var_mask * valid_mask
    L_c = np.sqrt(var_mask)
    return L_c

#############################################################

def func_cmlx_num_compute(img):
    r = 1
    nb = r*8 # neighborhood size
    otr = 6 # threshold for judging similar orientaion
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]]) / 3
    ky = np.transpose(kx)
    # Angle step. (coordinate)
    sps = np.zeros((nb, 2))
    at = 2 * np.pi/nb
    index = np.arange(nb)
    sps[:, 0] = -r*np.sin(index*at)
    sps[:, 1] = r*np.cos(index*at)
    # osvp computation
    imgd = np.pad(img, ((r, r), (r, r)), 'symmetric')
    h, w = imgd.shape
    Gx = conv(imgd, kx)  # gradient along x
    Gy = conv(imgd, ky)  # gradient along x
    Cimg = np.sqrt(np.power(Gx, 2.0) + np.power(Gy, 2.0)) # gradient
    Cvimg = np.zeros_like(imgd) # valid pixels (unshooth region)
    Cvimg[Cimg>=5] = 1 # selecting unshooth region # NOTE: result little different from matlab
    Oimg = np.round(np.arctan2(Gy, Gx)/np.pi*180 + eps) # algle value # NOTE: result little different from matlab

    Oimg[Oimg > 90] = Oimg[Oimg > 90] - 180 # [-90 90]
    Oimg[Oimg < -90] = 180 + Oimg[Oimg < -90]
    Oimg = Oimg + 90  # [ 0 180 ]
    Oimg[Cvimg==0] = 180+2*otr
    Oimgc = Oimg[r:-r, r:-r]
    Cvimgc = Cvimg[r:-r, r:-r]

    Oimg_norm = np.round(Oimg/2/otr + eps) # normalize with threshold 2*otr
    Oimgc_norm = np.round(Oimgc/2/otr + eps)
    onum = int(np.round(180/2/otr) + 1 + eps) # the type number of orientation bin
    # orientation types
    ssr_val = np.zeros((h-2*r, w-2*r, onum+1))
    for i in range(onum+1): # for central pixel
        Oimgc_valid = Oimgc_norm==i
        ssr_val[:, :, i] = ssr_val[:, :, i] + Oimgc_valid # the ori. no. on the x-th bin for each pixel
    for i in range(nb): # for neighbor pixels
        dx = int(np.round(r+sps[i,0]) + eps)
        dy = int(np.round(r+sps[i,1]) + eps)
        Oimgn = Oimg_norm[dx:h-2*r+dx, dy:w-2*r+dy]
        for j in range(onum+1):
            Oimg_valid = Oimgn==j
            ssr_val[:, :, j] = ssr_val[:, :, j] + Oimg_valid
    # complexity
    ssr_no_zero = ssr_val!=0
    cmlx = np.sum(ssr_no_zero, axis=2) # calculate the rule number
    cmlx[Cvimgc==0] = 1 # set the rule number of plain as 1
    cmlx[:r, :] = 1 # set the rule number for the image boundary
    cmlx[-r:, :] = 1
    cmlx[:, :r] = 1
    cmlx[:, -r:] = 1
    return cmlx


def func_ori_cmlx_compute(img):
    # compute the complexity value of each pixel with its osvp (Equ. 6)
    cmlx_map = func_cmlx_num_compute(img)
    r = 3
    sig = 1
    fker = gkern(r, sig)
    cmlx_mat = conv(cmlx_map, fker)
    return cmlx_mat

###############################################################
def func_randnum(col, row):
    #  create a matrix with the value 1 or -1
    randmat = np.random.rand(col, row)
    one = np.ones_like(randmat)
    minus_one = -np.ones_like(randmat)
    randmat = np.where(randmat > 0.5, one, minus_one)
    return randmat


def to_uint8(img):
    min_, max_ = np.min(img), np.max(img)
    img = (img - min_) / (max_ - min_) * 255
    img = img.astype(np.uint8)
    return img
##################################################

img = Image.open("lena.png")
img = np.array(img, dtype=np.float32)

# luminance adaptation
jnd_LA = func_bg_lum_jnd(img) # Equ. 13

# luminance contrast masking
L_c = func_luminance_contrast(img) # Equ. 7
alpha = 0.115*16
beta = 26
jnd_LC = (alpha*np.power(L_c, 2.4)) / (np.power(L_c, 2)+beta**2) # Equ. 11

# content complexity
P_c = func_ori_cmlx_compute(img) # Equ. 6
a1 = 0.3
a2 = 2.7
a3 = 1
C_t = (a1* np.power(P_c, a2)) / (np.power(P_c, 2)+a3**2) # Equ. 10


# pattern maksing
# L_c = np.log2(1+L_c)
jnd_PM = L_c * C_t # Equ. 8 # NOTE: different from the paper

# edge protection
edge_protect = func_edge_protect(img)
jnd_PM_p = jnd_PM * edge_protect

# visual masking
jnd_VM = np.where(jnd_LC>jnd_PM_p, jnd_LC, jnd_PM_p) # Equ. 12

# JND map
jnd_map = jnd_LA + jnd_VM - 0.3*np.where(jnd_LA<jnd_VM, jnd_LA, jnd_VM) # Equ. 14
jnd_map_temp = to_uint8(jnd_map)
out = Image.fromarray(jnd_map_temp)
out.save("JND_map.png")

# inject noise into image with the guidance of JND
randmat = func_randnum(*img.shape)
adjuster = 0.7
img_jnd = np.clip(img + adjuster * randmat * jnd_map, 0, 255)
img_jnd = img_jnd.astype(np.uint8)
out = Image.fromarray(img_jnd)
out.save("JND_img.png")
MSE_val = np.mean(np.square(img_jnd - img))
print('MSE = %.3f' % MSE_val)