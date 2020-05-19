import re
import spectral.io.envi as envi
import numpy as np
import geopandas
import ogr
import osr


dtype_map = [(1, np.uint8),                   # unsigned byte
             (2, np.int16),                   # 16-bit int
             (3, np.int32),                   # 32-bit int
             (4, np.float32),                 # 32-bit float
             (5, np.float64),                 # 64-bit float
             (6, np.complex64),               # 2x32-bit complex
             (9, np.complex128),              # 2x64-bit complex
             (12, np.uint16),                 # 16-bit unsigned int
             (13, np.uint32),                 # 32-bit unsigned int
             (14, np.int64),                  # 64-bit int
             (15, np.uint64)]                 # 64-bit unsigned int
envi_to_dtype = dict((k, v) for (k, v) in dtype_map)
dtype_to_envi = dict((v, k) for (k, v) in dtype_map)


def read_envi_header(hdr,
                     format_iter=None):

    f = open(hdr)
    hdr = f.read()

    # get all "[key] = " type matches starting at a new line
    # note that several white spaces may occur between [key] and =
    regex = re.compile(r'^.+?\s=\s', re.M | re.I)
    keys = regex.findall(hdr)

    # remove all new lines from header
    hdr = hdr.replace('\n', '')

    # remove 'ENVI' at the start of the header
    hdr = hdr[4:]

    # split header using keys in reverse order
    md_rev = dict()

    for key in keys[::-1]:

        # extract the corresponding item from the header
        item = hdr.split(key)[-1].strip()
        hdr = hdr.split(key)[0].strip()

        # clean up the key string
        key = key.strip()
        key = key.replace('=', '')
        key = key.strip()

        # add to the metadata dictionary
        md_rev[key] = item

    # although dictionary item order is arbitrary, sort items in 'correct' order anyway
    md = dict()
    keys = list(md_rev.keys())

    for key in keys[::-1]:

        md[key] = md_rev[key]

    # format mandatory integer items
    md['lines'] = int(md['lines'])
    md['samples'] = int(md['samples'])
    md['bands'] = int(md['bands'])
    md['data type'] = int(md['data type'])
    md['byte order'] = int(md['byte order'])
    md['header offset'] = int(md['header offset'])

    # format iterables of form "{a, b, c, ...}"
    # check for the following common iterables
    # additional iterables can optionally be included
    common_iter = [
        ['map info', str],  # most items of map info are numbers
        ['wavelength', float],
        ['fwhm', float],
        ['band names', str],
        ['spectra names', str],
        ['latitude', float],
        ['longitude', float]]

    if not format_iter:
        format_iter = common_iter
    else:
        format_iter = common_iter + list(format_iter)

    for f in format_iter:

        # get iterable key and desired data type
        key, data_type = f

        # format as list with items in desired data type
        if key in keys:

            item = md[key]
            if not (item == 'None' or item == 'NaN'):
                item = item[1:-1]
                item = item.split(',')
                item = [i.strip() for i in item]
                item = [data_type(i) for i in item]
                md[key] = np.array(item)

    return md


def read_envi_library(path):

    # add .hdr to path, remove other file extensions if needed
    if '.' in path:
        path = path.split('.')[0] + '.hdr'
    else:
        path += '.hdr'

    # read the spectra
    library = envi.open(path)
    spectra = np.array(library.spectra)

    # read the header metadata
    metadata = read_envi_header(path)

    return spectra, metadata


def save_envi_library(path, spectra, metadata):

    # here the path must free of file extensions
    if '.' in path:
        path = path.split('.')[0]

    library = envi.SpectralLibrary(spectra, metadata, None)
    library.save(path)


def save_library_shapefile(path, ids, labels, x, y, srs):

    # convert the srs string (wkt format) or integer (EPSG code) to a spatial reference object
    osr_srs = osr.SpatialReference()
    if isinstance(srs, str):
        osr_srs.ImportFromWkt(srs)
    elif isinstance(srs, int):
        osr_srs.ImportFromEPSG(srs)

    # create the output point shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(path)
    layer = ds.CreateLayer(path,
                           geom_type=ogr.wkbPoint,
                           srs=osr_srs)

    # create the spectra id and name field
    field = ogr.FieldDefn('id', ogr.OFTInteger)
    layer.CreateField(field)
    field = ogr.FieldDefn('label', ogr.OFTString)
    layer.CreateField(field)

    # add data to shapefile
    feature_definition = layer.GetLayerDefn()

    for sid, label, xcoord, ycoord in zip(ids, labels, x, y):

        # create point geometry
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(xcoord, ycoord)

        # create point feature and set its geometry
        feature = ogr.Feature(feature_definition)
        feature.SetGeometry(point)

        # set the id and label fields for this feature
        feature.SetField('id', int(sid))
        feature.SetField('label', label)

        # add the feature to the layer
        layer.CreateFeature(feature)

    # flush to disk
    del ds


def read_envi_image(path, load=True):

    # add .hdr to path, remove other file extensions if needed
    if '.' in path:
        path = path.split('.')[0] + '.hdr'
    else:
        path += '.hdr'

    # read the header metadata
    metadata = read_envi_header(path)

    # read image data
    image = envi.open(path)
    if load:
        image = image.load()

    return image, metadata


def save_envi_image(path, image, metadata):

    # add .hdr to path, remove other file extensions if needed
    if '.' in path:
        path = path.split('.')[0] + '.hdr'
    else:
        path += '.hdr'

    spyfile = envi.create_image(path,
                                metadata=metadata,
                                force=True,
                                ext='.bsq',
                                interleave='bsq')

    mm = spyfile.open_memmap(writable=True)
    mm[:, :, :] = image
    del mm



