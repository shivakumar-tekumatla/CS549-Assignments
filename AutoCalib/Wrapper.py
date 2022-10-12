"""Estimating parameters of the camera like the focal length, 
distortion coefficients and principle point is called Camera Calibration. 
It is one of the most time consuming and important part of any computer vision 
research involving 3D geometry."""
import cv2
import numpy as np
import os
import glob
from scipy.optimize import minimize,least_squares

class AutoCalib:
    def __init__(self,k=np.zeros((2,1))) -> None:
        self.vij = lambda H,i,j : np.array([H[0][i-1]*H[0][j-1], H[0][i-1]*H[1][j-1] + H[1][i-1]*H[0][j-1],H[1][i-1]*H[1][j-1],
                    H[2][i-1]*H[0][j-1] + H[0][i-1]*H[2][j-1],H[2][i-1]*H[1][j-1] + H[1][i-1]*H[2][j-1],H[2][i-1]*H[2][j-1]])
        self.k = k

    def intrinsicParameters(self,Homography_matrices):
        V =[]
        for H in Homography_matrices:
            v12 = self.vij(H,1,2)
            v11 = self.vij(H,1,1)
            v22 = self.vij(H,2,2)
            V.append(v12)
            V.append(v11-v22)

        # Use the matirx V to find out b . Vb = 0 
        # Solution to this is simply  the eigenvector of V'V associated with 
        # the smallest eigenvalue (equivalently, the right singular vector of 
        # V associated with the smallest singular value)

        eig,vec = np.linalg.eig(np.dot(np.transpose(V),V))
        vec = np.transpose(vec)
        b11, b12, b22, b13, b23, b33 = vec[np.argmin(eig)]

        v0 = (b12*b13 - b11*b23)/(b11*b22 - b12**2)
        Lambda = b33 - (b13**2 + v0*(b12*b13 - b11*b23))/b11
        alpha = np.sqrt(Lambda/b11)
        beta = np.sqrt(Lambda*b11 /(b11*b22 - b12**2))
        gamma = -b12*(alpha**2)*beta/Lambda
        u0 = gamma*v0/beta -b13*(alpha**2)/Lambda

        K = np.array([[alpha, gamma, u0],
                    [0,     beta,  v0],
                    [0,     0,      1]])
        return K

    def extrinsicParameters(self,K,H):
        h1,h2,h3 = H.T # hi = [hi1,hi2,hi3]' 
        K_inv = np.linalg.inv(K)
        Lambda = np.linalg.norm(K_inv.dot(h1),ord =2 )
        r1 = Lambda*K_inv.dot(h1)
        r2 = Lambda*K_inv.dot(h2)
        r3 = np.cross(r1,r2)
        t = Lambda*K_inv.dot(h3)
        return np.stack((r1,r2,r3,t), axis=1)

    def geometricError(self,parameters, imgpoints, objpoints, Extrinsics):

        alpha, beta, gamma, u0, v0, k1, k2 = parameters
        K =np.array([[alpha, gamma, u0],
                    [0,     beta,  v0],
                    [0,     0,      1]])

        error = []
        for i,RT in enumerate(Extrinsics):    
            r1,r2,r3,t = np.transpose(RT)
            R = np.stack((r1,r2,r3), axis=1)
            t = t.reshape(-1,1)
            obj_i,img_i = objpoints[i], imgpoints[i] 
            img_i = np.column_stack((img_i[0],np.ones(len(img_i[0])).reshape(-1,1))) 
            img_i = np.array([img_i],dtype = np.float64)
            img_i, _ = cv2.projectPoints(img_i, R, t, K, (k1,k2,0,0)) 
            err = [] 
            for xi,yi in  zip(obj_i, img_i.squeeze()):
                err.append(np.linalg.norm(xi-yi, ord=2)) 
            error = np.hstack((error,np.sum(err)))
        return error

def main():
    """Initial parts of this script os based on : 
    https://learnopencv.com/camera-calibration-using-opencv/
    In the above link, Once the corners are identified in each input image, 
    an inbuilt function is used to calibrate the camera. But we need to write this part by ourselves"""
    calib = AutoCalib()
    CHECKERBOARD = (6,9) # Defining the dimensions of checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = [] # Creating vector to store vectors of 3D points for each checkerboard image
    imgpoints = [] # Creating vector to store vectors of 2D points for each checkerboard image
    Homography_matrices = [] # Contains all the homography matrices. 

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    # Extracting path of individual image stored in a given directory
    calib_imgs_path = "Calibration_Imgs/*.jpg"
    images = glob.glob(calib_imgs_path)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        """
        If desired number of corner are detected, we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
            H,_ = cv2.findHomography(objp,corners2)
            Homography_matrices.append(H)  #contains all the Homography matrices 
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        out_file = "Output/"+fname.split("/")[1]
        cv2.imwrite(out_file,img)
    cv2.destroyAllWindows()

    """
    Performing camera calibration by passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the  detected corners (imgpoints).
    cv2.calibrateCamera already does this automatically. But we need do our own implementation 

    """
    K_initial = calib.intrinsicParameters(Homography_matrices) # Initializing the intrinsic parameters
    print("Initial intrinsic parameters \n", K_initial)
    #Estimating initial extrinsic parameters
    Extrinsics_initial = []
    for i, H in enumerate(Homography_matrices):
        RT = calib.extrinsicParameters(K_initial,H)
        Extrinsics_initial.append(RT) 
    
    # Now we need to solve the optimization problem to minimize the error 
    alpha = K_initial[0][0]
    beta = K_initial[1][1]
    u0 = K_initial[0][2]
    v0 = K_initial[1][2]
    k1,k2 = calib.k.ravel()
    gamma = K_initial[0][1]

    initial_parameters = [alpha, beta, gamma, u0, v0, k1, k2]

    initial_error = calib.geometricError(initial_parameters,imgpoints,objpoints, Extrinsics_initial)
    print("Initial Errors " , initial_error) # This is initial error . The goal is to reduce this . We get separate error for each image 
    optimal_parameters = least_squares(fun = calib.geometricError, x0 = initial_parameters, method="lm", args = [imgpoints,objpoints, Extrinsics_initial])

    #Now parameters are optimized  . Now finding the new intrinsic and extrinsic parameters 
    alpha, beta, gamma, u0, v0, k1, k2 = optimal_parameters.x 
    final_parameters = [alpha, beta, gamma, u0, v0, k1, k2]
    K_new = np.array([[alpha, gamma, u0],
                    [0,     beta,  v0],
                    [0,     0,      1]])
    print("Final intrinsic parameters \n", K_new)

    Extrinsics_new= []
    for i, H in enumerate(Homography_matrices):
        RT = calib.extrinsicParameters(K_new,H)
        Extrinsics_new.append(RT) 
    
if __name__ == "__main__":
    main()