from pyexiv2 import Image
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