import numpy as np
import scipy.signal as sp
import scipy.fftpack as sft
#import functions
import cv2

N = 255

RGB = np.matrix([[0.299,     0.587,     0.114],
                 [-0.16864, -0.33107,   0.49970],
                 [0.499813, -0.418531, -0.081282]])
YCbCr = RGB.I

def Rgb2Ycbcr(frame):
    """
    """
    xframe=np.zeros(frame.shape)
    for i in range(frame.shape[0]):
        xframe[i] = np.dot(RGB,frame[i].T).T
    return xframe

def filterFrame(frame, kernel):
    """
    """
    frame[:, :, 1] = sp.convolve2d(frame[:, :, 1], kernel, mode='same')
    frame[:, :, 2] = sp.convolve2d(frame[:, :, 2], kernel, mode='same')
    return frame


def applyDCT(frame, factor=4):
    
    r,c = frame.shape
    '''Make an array of 8 '1's '''
    Mr = np.ones(8) 
    '''Let only factor number of 1's and set the rest to zero'''
    Mr[factor:r] = np.zeros(8 - factor)
#     
    '''Block size is a square matrix 8x8 so Mc is same as Mr'''
    Mc = Mr
#     
    '''reshape the frame by columns of 8 and corresponding number of rows.
    This will help in Applying DCT to each row which has 8 values now'''
    
    frame=np.reshape(frame,(-1,8), order='C')
    
    X=sft.dct(frame,axis=1,norm='ortho')
    #apply row filter to each row by matrix multiplication with Mr as a diagonal matrix from the right:
    X=np.dot(X,np.diag(Mr))
    #shape it back to original shape:
    X=np.reshape(X,(-1,c), order='C')
    #Shape frame with columns of height 8 by using transposition .T:
    X=np.reshape(X.T,(-1,8), order='C')
    X=sft.dct(X,axis=1,norm='ortho')
    #apply column filter to each row by matrix multiplication with Mc as a diagonal matrix from the right:
    X=np.dot(X,np.diag(Mc))
    #shape it back to original shape:
    X=(np.reshape(X,(-1,r), order='C')).T
    #Set to zero the 7/8 highest spacial frequencies in each direction:
    #X=X*M    
    return X

def removeZeros(x, f):
    """
    """
    r,c = x.shape
    rem = 8 - f
    for k in range(f, c+1, f):
        x = np.delete(x, np.arange(k, k+rem),  axis=1)    
    for l in range(f, r+1, f):
        x = np.delete(x, np.arange(l, l+rem).T, axis=0)
    return x


framevectors = np.zeros((480, 640, 3))


def motionvector(Y, Yprev, blocksize):
    """
    """
    mv = np.zeros((60,80,2))
    block = np.array([8, 8])
    for yblock in range(blocksize):
    # print("yblock=",yblock)
        block[0] = yblock * 8+8;#150
        for xblock in range(blocksize):
            # print("xblock=",xblock)
            block[1] = xblock * 8+8;#200
            # print("block= ", block)
            # current block:
            Yc = Y[block[0]:block[0] + 8, block[1]:block[1] + 8]
            # print("Yc= ",Yc)
            # previous block:
            # Yp=Yprev[block[0]-4 : block[0]+12 ,block[1]-4 : block[1]+12]
            # print("Yp= ",Yp)
            # correlate both to find motion vector
            # print("Yp=",Yp)
            # print(Yc.shape)
            # Some high value for MAE for initialization:
            bestmae = 100.0;
            # For loops for the motion vector, full search at +-8 integer pixels:
            # print "reset counter"
            ctr = 0;
            for ymv in range(-8, 8):
                for xmv in range(-8, 8):
                    diff = Yc - Yprev[block[0] + ymv: block[0] + ymv + 8, block[1] + xmv: block[1] + xmv + 8];
                    mae = sum(sum(np.abs(diff))) / 64;
                    ctr = ctr + 1;
    
                    # print "ctr: ", ctr
    
                    if mae <= bestmae:
                        # print "mae: ", mae
                        bestmae = mae;
                        mv[yblock, xblock, 0] = ymv
                        mv[yblock, xblock, 1] = xmv
    
                    if mae < 1:
                        ctr = 0
                        # print "counter break"
                        break
     
            if bestmae > 10:
                # print "Motion Detected . . . . . . . . . . . . "
                cv2.line(framevectors, (block[1], block[0]), (block[1] + mv[yblock, yblock, 1].astype(int), block[0] + mv[yblock, yblock, 0].astype(int)),
                         (1.0, 1.0, 1.0));
            elif bestmae < 10:
                # print "bestmae", bestmae
                # print "No motion detected  . . . . . . . . . . "
                cv2.line(framevectors, (block[1], block[0]), (block[1], block[0]), (1.0, 1.0, 1.0))
    return mv




def fillZeros(frame,flag):
    """
    """
    if flag == 1:
        r, c = frame.shape
        factor = r//60 #or c/80
        incRC = 480.//r
        rem = 8-factor
    else:
        r, c = frame.shape
        factor = r//30 #or c/80
        incRC = 240.//r
        rem = 8-factor
    for i in range(factor, int(r*incRC), 8):
        frame = np.insert(frame, [i], np.zeros(rem).reshape(rem,1), axis=0)
    for i in range(factor, int(c*incRC), 8):
        frame = np.insert(frame,[i], np.zeros(rem), axis=1)
    return frame


def applyIDCT(frame):
    """
    """
    r,c= frame.shape
    X=np.reshape(frame,(-1,8), order='C')
    X=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    X=np.reshape(X,(-1,c), order='C')
    #Shape frame with columns of hight 8 (columns: order='F' convention):
    X=np.reshape(X.T,(-1,8), order='C')
    x=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    x=(np.reshape(x,(-1,r), order='C')).T
    return x

def Ycbcr2Rgb(frame):
    """
    """
    xframe=np.zeros(frame.shape)
    for i in range(frame.shape[0]):
        xframe[i] = np.dot(YCbCr, frame[i].T).T#/255.
    return xframe



def inverse_motionvector(mv, Ybuff):
    #print "Entered decoder odd"
    Yc = Ybuff.copy()
    #             print mv[:,:,0]
    for i in range(1,mv.shape[0]):
        for j in range(1,mv.shape[1]):
                            
            if mv[i,j,0] != 0 or mv[i,j,1] != 0:

                new_pos = mv[i,j,:].astype(np.int) + np.array([i*8+8, j*8+8])
#                         print( "new pos", new_pos)
                old_pos = np.array([i*8+8, j*8+8])
#                         print( "old_pos", old_pos)
                
                if new_pos[0] > 4 and new_pos[1] > 4:
                    Yc[old_pos[0]-4: old_pos[0]+4, old_pos[1]-4: old_pos[1]+4] = Ybuff[new_pos[0]-4:new_pos[0]+4, new_pos[1]-4:new_pos[1]+4]

    return Yc