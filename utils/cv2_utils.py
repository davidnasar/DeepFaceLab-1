import cv2
import numpy as np
from pathlib import Path

#allows to open non-english characters path
def cv2_imread(filename, flags=cv2.IMREAD_UNCHANGED):
    try:
        with open(filename, "rb") as stream:
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            return cv2.imdecode(numpyarray, flags)
    except:
        return None

def cv2_imwrite(filename, img, *args):
    ret, buf = cv2.imencode( Path(filename).suffix, img, *args)
    if ret == True:
        try:
            with open(filename, "wb") as stream:
                stream.write( buf )
        except:
            pass
