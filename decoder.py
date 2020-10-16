import cv2
import pickle
import methods
import numpy as np
import scipy.signal as sp
r,c,d = 480,640,3

g = open('compressed.txt', 'rb')
ctr = 0

N = 2
lpF = np.ones((N, N))/N
pyF = sp.convolve2d(lpF, lpF)/N

Ybuff = np.zeros((r,c))

try:
    while True:

        if ctr % 2 == 0:

            yz = pickle.load(g)
            uz = pickle.load(g)
            vz = pickle.load(g)
    
            ydctdec = methods.fillZeros(yz, 1)
            udctdec = methods.fillZeros(uz, 0)
            vdctdec = methods.fillZeros(vz, 0)
    
            ydec = methods.applyIDCT(ydctdec)
            udec = methods.applyIDCT(udctdec)
            vdec = methods.applyIDCT(vdctdec)
            #ydec, ydec, vdec = methods.idctFrame([ydctdec, udctdec, vdctdec])
            
            # upsample
        
            udecu = np.zeros((r,c))
            vdecu = np.zeros((r,c))
            
            udecu[::2, ::2] = udec
            vdecu[::2, ::2] = vdec
            
    #         print vdecu.shape
       
            framedec = np.zeros((r,c,d))
            framedec[:,:,0] = ydec
            framedec[:,:,1] = udecu
            framedec[:,:,2] = vdecu
            
            framedecFilt = methods.filterFrame(framedec, lpF)
            
            decRGB = methods.Ycbcr2Rgb(framedecFilt)
            
            cv2.imshow("Frame without Motion Compensation", decRGB) 
        
        else:
            "pickle load"
            mv = pickle.load(g)
            Yc = methods.inverse_motionvector(mv, Ybuff)
            framedec[:,:,0] = Yc
            framedec[:,:,1] = framedecFilt[:,:,1]
            framedec[:,:,2] = framedecFilt[:,:,2]
            
            decRGB = methods.Ycbcr2Rgb(framedec)
            cv2.imshow('Frame With Motion Compensation', decRGB)
        #cv2.imshow('Decoded', decRGB)                
        Ybuff = framedecFilt[:,:,0].copy()
        ctr += 1
        
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
except(EOFError):
    pass
cv2.destroyAllWindows()
g.close()
        
        
        
