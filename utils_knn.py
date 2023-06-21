# from pyexiv2 import Image
# from WaterFowlTools.utils import py_cpu_nms, get_image_taking_conditions, get_sub_image

def read_LatLotAlt(image_dir):
    info = Image(image_dir)
    exif_info = info.read_exif()
    xmp_info = info.read_xmp()
    re = dict()
    re['latitude'] = float(xmp_info['Xmp.drone-dji.GpsLatitude'])
    re['longitude'] = float(xmp_info['Xmp.drone-dji.GpsLongitude'])
    re['altitude'] = float(xmp_info['Xmp.drone-dji.RelativeAltitude'][1:])
    #print (image_name,xmp_info['Xmp.drone-dji.RelativeAltitude'])
    return re
def get_GSD(altitude,camera_type='Pro2', ref_altitude=60):

    if (camera_type == 'Pro2'):
        ref_GSD = (13.2 * ref_altitude)/(10.26*5472)
        GSD = (13.2 * altitude)/(10.26*5472)
    elif (camera_type == 'Air2'):
        ref_GSD = (6.4*ref_altitude)/(4.3*8000)
        GSD = (6.4*altitude)/(4.3*8000)
    else:
        ref_GSD = (13.2 * ref_altitude)/(10.26*5472)
        GSD = (13.2 * altitude)/(10.26*5472)
    return GSD, ref_GSD

def filter_slice(bbox,coor,sub_image_width,mega_image_shape,dis = 5):
    """
    Merging bbox from slices ver1.
    Hide the box that are connected to the edges
    """ 
    re = []

    boundary = [[coor[0],coor[1]],[coor[0]+sub_image_width,coor[1]+sub_image_width]]
    thresh = [[0,0],[512,512]]
    if (boundary[0][0]==0):
        thresh[0][0]-=100
    if (boundary[0][1]==0):
        thresh[0][1]-=100
    if (boundary[1][0]==mega_image_shape[0]):
        thresh[1][0]+=100
    if (boundary[1][1]==mega_image_shape[1]):
        thresh[1][1]+=100
    #print (boundary,mega_image_shape)
    for box in bbox:
        center = [(box[0]+box[2])//2,(box[1]+box[3])//2]
        if (center[1]-dis<=thresh[0][0] or center[0]-dis<=thresh[0][1]):
            continue
        if (center[1]+dis>=thresh[1][0] or center[0]+dis>=thresh[1][1]):
            continue
        re.append(box)

    return re


if __name__ == '__main__':
    re = union_slice_box([[],[]],[(0,0),(0,512)],512,(512,1024))
