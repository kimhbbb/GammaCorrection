import numpy as np
import cv2
import matplotlib.pyplot as plt

class GammaCorrector:
    def __init__(self, threshold = 0.5, usingMedian = False):
        self.threshold = threshold
        self.usingMedian = usingMedian
    
    def get_image(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)


    #2.1 Luminance normalization
    def luminance_normalization(self):
        self.Y_i = 0.299 * self.image[:, :, 2] + 0.587 * self.image[:, :, 1] + 0.114 * self.image[:, :, 0]

        M = np.max(self.Y_i)
        E = 1
        self.Y_l = np.log(self.Y_i + E) / np.log(M+E)

        print(self.Y_l)


    # 2.2 Optimal gamma correction parameter estimation
    def newton_method_average(self, s, sigma, gamma_init, max_iter=100, min_gamma=0, max_gamma=1):
        gamma = gamma_init
        E = 1e-7
        for i in range(max_iter):
            phi_s = np.power(s, gamma)

            numerator = np.average(phi_s) - sigma
            denominator = np.average(phi_s * np.log(s + E))
            
            print(f"iter {i+1}")
            print(f"divide: {(numerator / denominator):.8f} / numerator: {numerator:.3f} / denominator: {denominator:.3f}")

            gamma_new = gamma - numerator / denominator
            gamma_new = np.clip(gamma_new, min_gamma, max_gamma)

            print(f"-> newgamma {gamma_new:.3f} = {gamma:.3f} - {(numerator / denominator):.3f}\n")
            
            if np.abs(gamma_new - gamma) < E:
                break
            
            gamma = gamma_new

        return gamma, i+1
    
    def newton_method_median(self, s, sigma, gamma_init, max_iter=101, min_gamma=0, max_gamma=1):
        gamma = gamma_init
        E = 1e-7
        for i in range(max_iter):
            phi_s = np.power(s, gamma)

            numerator = np.median(phi_s) - sigma
            denominator = np.average(phi_s * np.log(s + E))
            
            print(f"iter {i+1}")
            print(f"divide: {(numerator / denominator):.8f} / numerator: {numerator:.3f} / denominator: {denominator:.3f}")

            gamma_next = gamma - numerator / denominator
            gamma_next = np.clip(gamma_next, min_gamma, max_gamma)

            print(f"-> newgamma {gamma_next:.3f} = {gamma:.3f} - {(numerator / denominator):.3f}\n")
            
            if np.abs(gamma_next - gamma) < E:
                break
            
            gamma = gamma_next

        return gamma, i+1

    def estimate_optimal_gamma(self):
        dark_region_mask = self.Y_l <= self.threshold
        bright_region_mask = self.Y_l > self.threshold

        s_d = self.Y_l[dark_region_mask]
        sigma_d = np.std(s_d) 
        N_d = len(s_d)

        s_b = self.Y_l[bright_region_mask]
        sigma_b = 1 - sigma_d
        N_b = len(s_b)

        self.nd_ratio = N_d / (N_d+N_b)
        self.sigma_d = sigma_d


        print("dark------------------------------------------------")
        if self.usingMedian:
            self.gamma_d, iter_d = self.newton_method_median(s_d, sigma_d, gamma_init = 1, max_iter=100, min_gamma = 0, max_gamma = 1)
            print("bright------------------------------------------------")
            self.gamma_b, iter_b = self.newton_method_median(s_b, sigma_b, gamma_init = 1, max_iter=100, min_gamma = 1, max_gamma = 10)
            print("newton method: median")
        else:
            self.gamma_d, iter_d = self.newton_method_average(s_d, sigma_d, gamma_init = 1, max_iter=100, min_gamma = 0, max_gamma = 1)
            print("bright------------------------------------------------")
            self.gamma_b, iter_b = self.newton_method_average(s_b, sigma_b, gamma_init = 1, max_iter=100, min_gamma = 1, max_gamma = 10)
            print("newton method: average")
    
        print(f"dark gamma(iter {iter_d}): {self.gamma_d:.8f}, bright gamma(iter {iter_b}): {self.gamma_b:.8f}")
        

    # 2.3 Fusion of corrected images
    def fusion_images(self):
        Y_d_corrected = np.power(self.Y_l, self.gamma_d)
        Y_b_corrected = np.power(self.Y_l, self.gamma_b)

        sigma_w = 0.5
        w = np.exp(-np.power(Y_b_corrected, 2) / (2 * sigma_w ** 2))
        self.Y_o = w * Y_d_corrected + (1 - w) * Y_b_corrected

        print("Y_o dtype: {}".format(self.Y_o.dtype))
        

    # 2.4 Adaptive color restoration
    def adaptive_color_restoration(self):
        I_i_R = self.image[:, :, 2]
        I_i_G = self.image[:, :, 1]
        I_i_B = self.image[:, :, 0]

        E = 1
        s = 1 - np.tanh(self.Y_l)
        I_o_R = self.Y_o * (np.power((I_i_R + E) / (self.Y_i + E), s))
        I_o_G = self.Y_o * (np.power((I_i_G + E) / (self.Y_i + E), s))
        I_o_B = self.Y_o * (np.power((I_i_B + E) / (self.Y_i + E), s))

        self.I_o = np.stack([I_o_B, I_o_G, I_o_R], axis=-1)
        print("OUTPUT : {}".format(self.I_o.shape))

    def show_image(self):
        print(f"ratio of Dark Region: {self.nd_ratio:.3f}")
        print(f"sigma_d: {self.sigma_d:.3f}\n")

        cv2.imshow("Original", self.image.astype(np.uint8))

        method = "average"
        if self.usingMedian:
            method = "median"
        print("DTYPE: {}".format(self.I_o.dtype))
        cv2.imshow(f"Result with {method}", self.I_o)
        cv2.waitKey(0)

    def gamma_correction(self, image_path):
        self.__init__(self.threshold, self.usingMedian)
        self.get_image(image_path)
        
        self.luminance_normalization()
        self.estimate_optimal_gamma()
        self.fusion_images()
        self.adaptive_color_restoration()
        self.show_image()        
        print(f"----------------complete gamma correction {image_path}--------------------\n")


image_paths = [ 
                "2015_07240.png",
                #"c:\\Users\\USER\\Desktop\\test\\archive\\Table\\2015_07240.png",
                #"c:\\Users\\USER\\Desktop\\test\\archive\\Bus\\2015_01884.png",
                #"c:\\Users\\USER\\Desktop\\test\\archive\\Bus\\2015_01882.png",
                #"c:\\Users\\USER\\Desktop\\test\\archive\\Car\\2015_02725.png",
              ]

image_paths2 = [ 
                #"c:\\Users\\USER\\Desktop\\test\\images\\3.jpg",
                #"c:\\Users\\USER\\Desktop\\test\\images\\d.png",
                #"c:\\Users\\USER\\Desktop\\test\\images\\4.jpg",
                #"c:\\Users\\USER\\Desktop\\test\\images\\q.jpg",
                #"c:\\Users\\USER\\Desktop\\test\\images\\t.jpg",
                #"c:\\Users\\USER\\Desktop\\test\\images\\y.jpg",
                #"c:\\Users\\USER\\Desktop\\test\\images\\r.jpg",
              ]

gamma_corrector = GammaCorrector(threshold = 0.5, usingMedian = False)
for path in image_paths:
    gamma_corrector.gamma_correction(path)

cv2.destroyAllWindows()