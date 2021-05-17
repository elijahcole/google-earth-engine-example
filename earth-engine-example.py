import ee
import time
import numpy as np
import time
import sys
import json
import argparse

def UTM_lookup(lon):
    assert (lon < 180.0) and (lon >= -180.0)
    # assuming lon in [-180,180)
    return int(np.floor((lon + 180.0)/6) + 1)

def EPSG_lookup(lon,lat):
    utm_zone = UTM_lookup(lon)
    if lat >= 0:
        return 32600 + utm_zone
    else:
        return 32700 + utm_zone

def get_box(lon, lat, patch_extent_m):
    '''
    First selects the right extent using UTM coordinates, then resamples. 
    '''
    query_point = ee.Geometry.Point([lon, lat], crs='EPSG:4326')
    ctr_coords = query_point.getInfo()['coordinates'] # [lon, lat]
    crs_local = 'EPSG:' + str(EPSG_lookup(lon, lat)) # convert to UTM
    ctr_coords_local = query_point.transform(crs_local).getInfo()['coordinates'] # [easting, northing]
    offset = patch_extent_m / 2.0
    ur_coords_local = [ctr_coords_local[0] + offset, ctr_coords_local[1] + offset]
    ll_coords_local = [ctr_coords_local[0] - offset, ctr_coords_local[1] - offset]
    ur = ee.Geometry.Point(ur_coords_local, crs_local).transform('EPSG:4326').getInfo()['coordinates']
    ll = ee.Geometry.Point(ll_coords_local, crs_local).transform('EPSG:4326').getInfo()['coordinates']
    sel_box = ee.Geometry.Rectangle([ll[0], ll[1], ur[0], ur[1]], 'EPSG:4326')
    return sel_box

def export_landsat_series(lon, lat, patch_extent, image_size, fname_base):
    '''
    Exports a time series of (raw) Landsat-8 imagery for a given location. 
    '''
    
    # define series parameters: 
    dataset = 'LANDSAT/LC08/C01/T1_SR' # note: must modify this function if dataset changed
    date_start = '2013-01-01'
    date_end = '2019-12-31'
    
    # get object defining region of interest: 
    query_box = get_box(lon, lat, patch_extent)

    # filter image collection down to those that interect region and dates of interest: 
    ds = ee.ImageCollection(dataset)
    ds = ds.filterBounds(query_box)
    ds = ds.filterDate(date_start,date_end)
    
    # convert image collection to list, sorted by date: 
    num_images = ds.size().getInfo()
    ds_list = ds.sort('system:time_start', True).toList(num_images)
    print('found %d matching images'%(num_images))
    
    task_dict = {}
    max_tasks = 2000 # server-side cap is 3000
    
    for i in range(num_images):
        
        # select current image object: 
        I = ee.Image(ds_list.get(i))
        
        # extract raster of pixel quality attributes:
        I_pixel_qa = I.select('pixel_qa')
        
        # extract raster of radiometric saturation quality attributes: 
        I_radsat_qa = I.select('radsat_qa')
        
        # extract raster of multispectral imagery: 
        I_multispectral = I.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11'])

        # generate file name: 
        date_str = str(ee.Date(I.get('system:time_start')).format('YYYY-MM-DD').getInfo())
        idx_str = str(i).zfill(5)
        cur_fname = fname_base + '_' + idx_str + '_' + date_str
        print(cur_fname)
        
        # generate other export parameters: 
        export_size = str(image_size)+'x'+str(image_size)
        export_crs = 'EPSG:'+str(EPSG_lookup(lon, lat))
        export_region = query_box.getInfo()['coordinates']
        
        # export multispectral imagery: 
        task_dict[str(i)+'_multispectral'] = ee.batch.Export.image.toDrive(
            image=I_multispectral,
            region=export_region,
            dimensions=export_size,
            crs=export_crs,
            description=cur_fname+'_multispectral',
            folder=fname_base
            )
        task_dict[str(i)+'_multispectral'].start()
        if i == 0:
            time.sleep(30) # pause to allow time for the first to complete - otherwise can get duplicate folders. 
        
        # export pixel quality attributes: 
        task_dict[str(i)+'_pixelqa'] = ee.batch.Export.image.toDrive(
            image=I_pixel_qa,
            region=export_region,
            dimensions=export_size,
            crs=export_crs,
            description=cur_fname+'_pixelqa',
            folder=fname_base
            )
        task_dict[str(i)+'_pixelqa'].start()

        # export radiometric saturation quality attributes: 
        task_dict[str(i)+'_radsatqa'] = ee.batch.Export.image.toDrive(
            image=I_radsat_qa,
            region=export_region,
            dimensions=export_size,
            crs=export_crs,
            description=cur_fname+'_radsatqa',
            folder=fname_base
            )
        task_dict[str(i)+'_radsatqa'].start()
        
        # determine the number of active tasks: 
        num_active = 0
        done_list = []
        for t in task_dict:
            is_active = int(task_dict[t].status()['state'] in ['READY', 'RUNNING'])
            if is_active:
                num_active += 1
            else:
                done_list.append(t)
        print('%d active tasks'%(num_active))
                
        # remove completed jobs from the task dictionary: 
        for t in done_list:
            del task_dict[t]
        
        # wait if too many tasks are active: 
        while num_active >= max_tasks:
            time.sleep(10)

'''
load and parse locations: 
'''

pp = argparse.ArgumentParser(description='PyTorch MultiLabel Train')
pp.add_argument('--path-to-file-names', type=str, default='', help='path to npy file containing list of N location names')
pp.add_argument('--path-to-longitudes', type=str, default='', help='path to npy file containing list of N longitude values')
pp.add_argument('--path-to-latitudes', type=str, default='', help='path to npy file containing list of N latitude values')
args = pp.parse_args()

patch_extent = int(6000) # meters
image_size = int(200) # pixels

loc_names = np.load(args.path_to_file_names) # N location names
lon_list = np.load(args.path_to_longitudes) # N longitudes
lat_list = np.load(args.path_to_latitudes) # N latitudes

# Note: all images (across time) for location (lon_list[i], lat_list[i]) are 
# saved into a folder named loc_names[i] in Google Drive. 

'''
run earth engine calls: 
'''

ee.Initialize()

assert len(lon_list) == len(lat_list)
    
for i in range(len(lon_list)):
    
    print('processing {}'.format(loc_names[i]))
        
    # export imagery:
    export_landsat_series(lon_list[i], lat_list[i], patch_extent, image_size, loc_names[i])

# Note: need to execute with -u option if logging isn't working with nohup. 
