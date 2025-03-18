import cv2
import numpy as np
import matplotlib.pyplot as plt

def luminance_normalize(Y_i):
    epsilon = 1
    # M = np.max(Y_i) + 1e-7
    M = np.max(Y_i) + epsilon # @@@ 이거 약간 애매하다. M이 Y의 max라고 했을 때, 1을 더하면 Y의 표현 범위를 넘는 거 아닌가?

    Y_L = (np.log(Y_i + epsilon) / np.log(M)).astype(np.float32)
    # print(np.any((Y_L < 0) | (Y_L > 1))) # False @@@ [0,1] 만족 -> clip 필요 없나? 근데 뒤에 분모, 분자 때문에 있어야 할 거 같기는 한데... 

    # Y_L = np.clip(Y_L, 1e-7, 1.0)
    # Y_L = np.clip(Y_L, 0.0, 1.0)

    return Y_L

def optimize_gamma(region_values, region_std, gamma_range, initial_gamma = 1.0, max_iter=1000, tolerance = 1e-7):
    gamma = initial_gamma

    region_values = np.average(region_values)
    
    for _ in range(max_iter):
        # numerator = np.average(np.power(region_values, gamma) - region_std)
        # numerator = np.average(np.power(region_values, gamma)) - region_std
        numerator = np.power(region_values, gamma) - region_std
        denominator = np.average(np.power(region_values, gamma) * np.log(region_values + 1e-7))

        gamma_new = gamma - (numerator / denominator)
        gamma_new = np.clip(gamma_new, gamma_range[0] + 1e-7, gamma_range[1])

        if np.abs(gamma_new - gamma) < tolerance:
            break
        
        gamma = gamma_new

    return gamma

def fusion_corrected_images(gamma_d, gamma_b):
    y_d = np.zeros_like(Y_L)
    y_b = np.zeros_like(Y_L)

    y_d = np.power(Y_L, gamma_d)
    y_b = np.power(Y_L, gamma_b)

    W_std = 0.5
    W = np.exp(- np.power(y_b, 2) / (2 * (W_std ** 2)))
    W = W.astype(np.float32)

    # print("W: {}".format(W))

    # 다른 부분
    y_o = W * np.power(y_d, gamma_d) + (1 - W) * np.power(y_b, gamma_b)
    y_o = np.clip(y_o, 0.0, 1.0)

    # print("Y_O: {}".format(y_o))

    return y_o, y_b, y_d

def test_adaptive_color_restoration():
    # 다른 부분
    s_bright = 1 - np.tanh(y_b)
    s_dark = np.tanh(y_d)
    s_bright = np.clip(s_bright, 1e-7, 1.0)
    s_dark = np.clip(s_dark, 1e-7, 1.0)


    
    output_color = []

    for color in (I_B, I_G, I_R):
        # restored = y_o * np.power(color / y_L, s)
        # I_o_color = y_o * np.power((color + 1e-7) / (Y_i + 1), s)

        y_d_mark = np.where(dark_region_mask, color, 0)
        y_b_mark = np.where(bright_region_mask, color, 0)

        I_o_color = y_o * (np.power((y_b_mark + 1) / (Y_i + 1), s_bright) + np.power((y_d_mark + 1) / (Y_i + 1), s_dark))
        I_o_color = np.clip(I_o_color, 0, 1)
        I_o_color = I_o_color.astype(np.float32)
        # print("color: {}, value: {}".format(color, I_o_color))
        output_color.append(I_o_color)

    # print("OUTPUTSHAPE : {}".format(output_color[0].shape))
    output_image_bgr = cv2.merge(output_color)
    # output_image_bgr = cv2.merge([output_color[0], output_color[1], output_color[2]])

    return output_image_bgr


# --- 2.0 ---
img = cv2.imread("2015_07240.png")

I_B, I_G, I_R = cv2.split(img) # dtype: uint8, range: 0~255

# --- 2.1 luminance normalize ---
Y_i = (0.299 * I_R + 0.587 * I_G + 0.114 * I_B).astype(np.float32) # dtype: float64 -> float32

Y_L = luminance_normalize(Y_i)


# --- 2.2 optimal gamma correction parameter esimation ---
threshold = 0.5

dark_region_mask = Y_L <= threshold
bright_region_mask = Y_L > threshold

print(bright_region_mask)
print(bright_region_mask.shape)

s_d = Y_L[dark_region_mask] # the vector of original pixel values in the dark range
s_b = Y_L[bright_region_mask] 

std_dark = np.std(s_d) # the standard deviation of pixel values in the dark range
std_bright= 1 - std_dark

gamma_dark = optimize_gamma(s_d, std_dark, initial_gamma=1.0, gamma_range=(1e-7, 1.0))
gamma_bright = optimize_gamma(s_b, std_bright, initial_gamma=10.0, gamma_range=(1.0, 10.0))

# print("gamma_dark: {}".format(optimal_gamma_dark))
# print("gamma_bright: {}".format(optimal_gamma_bright))


# --- Fusion of corrected images ---
y_o, y_b, y_d = fusion_corrected_images(gamma_dark, gamma_bright)


# --- Adaptive color restoration ---
output_img = test_adaptive_color_restoration()

cv2.imshow("each s Image",output_img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
