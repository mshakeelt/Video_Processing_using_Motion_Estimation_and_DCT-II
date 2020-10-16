import numpy as np
import cv2
import methods
import scipy.signal as sp
import pickle
import multiprocessing


compressed = open('Compressed.txt', 'wb')

original = open('original.txt', 'wb')

N=2

factor = 4
blocksize = 20
lpF = np.ones((N, N))/N
pyF = sp.convolve2d(lpF, lpF)/N

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

r, c, d = frame.shape

mv  = np.zeros((r//8,c//8, 2))

Yprev = np.zeros((r,c))

for i in range(25):
    
    ret, frame = cap.read()

    if ret == True:
        reduced = frame.copy()
        
        frameYUV = methods.Rgb2Ycbcr(reduced)

        if i%2 == 0:
            frameFilt = methods.filterFrame(frameYUV, pyF)
    
            Y = frameFilt[:,:,0]/255.0
            Cb = frameFilt[:,:,1]/128.0
            Cr = frameFilt[:,:,2]/128.0   
            
            Cbd = Cb[::2,::2]
            Crd = Cr[::2,::2]
            if i==0:
                print(Cb.shape, Cbd.shape) 
            ##DCT
            #ydct, udct, vdct = methods.dctFrame([Y, Cbd, Crd], factor)
            ydct = methods.applyDCT(Y)
            udct = methods.applyDCT(Cbd)
            vdct = methods.applyDCT(Crd)
            
            #if __name__ ==  '__main__':
             #   pool = multiprocessing.Pool(3)
              #  dct_frame = pool.map(methods.applyDCT, [Y, Cbd, Crd])
               # pool.close()
                #pool.join()
            # Remove zeros
            
            ydctwoz = methods.removeZeros(ydct, factor)
            udctwoz = methods.removeZeros(udct, factor)
            vdctwoz = methods.removeZeros(vdct, factor)
            "pickle dump dct"
            pickle.dump(ydctwoz.astype(np.float16), compressed)
            pickle.dump(udctwoz.astype(np.float16), compressed)
            pickle.dump(vdctwoz.astype(np.float16), compressed)
            
        else:
            Y = frameYUV[:,:,0]/255.0
            mv = methods.motionvector(Y, Yprev, blocksize)
            "pickle dump motion vectors"

            pickle.dump(mv, compressed)
        
        pickle.dump(frame, original)
        Yprev = Y.copy()

        cv2.imshow('Y',Y)# / 255 )
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
print ("Saved the captured video data and compressed video data into .txt files")
print ("Uncompressed file size")
print (original.tell()/(1024*1024) , "MBs")
print ("Compressed file size")
print (compressed.tell()/(1024*1024) , "MBs")

cap.release()
cv2.destroyAllWindows()  
compressed.close()  
original.close()