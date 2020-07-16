import copy

from gultools.io import *


def coord2rowcol(x, y, xul, yul, xres, yres):

    """
    transforms x-y coordinates to image row-column
    :param x: coordinate, float or iterable
    :param y: coordinate, float or iterable
    :param xul: x coordinate upper left corner
    :param yul: y coordinate upper left corner
    :param xres: x resolution
    :param yres: y resolution
    :return: row, col
    """

    col = np.floor((x - xul) / xres)
    row = np.floor((yul - y) / yres)

    if hasattr(col, '__iter__'):
        col = np.array(col, dtype=int)
        row = np.array(row, dtype=int)
    else:
        col = int(col)
        row = int(row)

    return row, col


def rowcol2coord(row, col, xul, yul, xres, yres):

    """
    transforms image row-col to x-y coordinates
    :param row: coordinate, float or iterable
    :param col: coordinate, float or iterable
    :param xul: x coordinate upper left corner
    :param yul: y coordinate upper left corner
    :param xres: x resolution
    :param yres: y resolution
    :return:
    """

    x = col * xres + xul + xres / 2
    y = yul - row * yres - yres / 2

    return x, y


def transform_coordinates(x_src, y_src, srs_src, srs_dst):

    """
    transforms x-y coordinates to different reference system
    :param x_src: coordinate, float or iterable
    :param y_src: coordinate, float or iterable
    :param srs_src: source spatial reference system, either EPSG code (int) or WKT (string)
    :param srs_dst: destination spatial reference system, either EPSG code (int) or WKT (string)
    :return: x_dst, y_dst
    """

    osr_srs_src = osr.SpatialReference()
    if isinstance(srs_src, str):
        osr_srs_src.ImportFromWkt(srs_src)
    elif isinstance(srs_src, int):
        osr_srs_src.ImportFromEPSG(srs_src)

    osr_srs_dst = osr.SpatialReference()
    if isinstance(srs_dst, str):
        osr_srs_dst.ImportFromWkt(srs_dst)
    elif isinstance(srs_dst, int):
        osr_srs_dst.ImportFromEPSG(srs_dst)

    transformer = osr.CoordinateTransformation(osr_srs_src, osr_srs_dst)
    x_dst = []
    y_dst = []

    for x, y in zip(x_src, y_src):

        xtemp, ytemp, ztemp = transformer.TransformPoint(x, y)
        x_dst.append(xtemp)
        y_dst.append(ytemp)

    x_dst = np.array(x_dst)
    y_dst = np.array(y_dst)

    return x_dst, y_dst


def tile_image(image_path, tile_size, outfolder, basename,
               remove_empty_tiles=True,
               nodata=None):

    """
    tiles an image in smaller square subsets
    :param image_path: str
    :param tile_size: int, pixels
    :param outfolder: str
    :param basename: str, image names = basename + counter
    :param remove_empty_tiles: bool
    :param nodata: int or float
    :return: None
    """

    image, metadata = read_envi_image(image_path, load=False)
    rows = metadata['lines']
    cols = metadata['samples']
    map_info = metadata['map info']
    ulx = float(map_info[3])
    uly = float(map_info[4])
    resx = float(map_info[5])
    resy = float(map_info[6])
    trows = int(np.ceil(rows / tile_size))
    tcols = int(np.ceil(cols / tile_size))
    ntiles = trows * tcols
    ndigits = len(str(ntiles))

    cnt = -1

    for r in range(trows):

        for c in range(tcols):

            image_tile = image[r * tile_size:(r + 1) * tile_size, c * tile_size:(c + 1) * tile_size, :]

            if remove_empty_tiles and np.all(image_tile == nodata, axis=None):

                pass

            else:

                metadata_tile = copy.deepcopy(metadata)
                metadata_tile['lines'] = image_tile.shape[0]
                metadata_tile['samples'] = image_tile.shape[1]
                ulx_tile = ulx + c * tile_size * resx
                uly_tile = uly - r * tile_size * resy
                metadata_tile['map info'][3] = ulx_tile
                metadata_tile['map info'][4] = uly_tile

                cnt += 1
                ndigits_cnt = len(str(cnt))
                name = basename + '_' + (ndigits - ndigits_cnt) * '0' + str(cnt)
                outpath = outfolder + r'\{}.bsq'.format(name)

                save_envi_image(outpath, image_tile, metadata_tile)


def merge_libraries(lib_paths, out_path,
                    source=None,
                    additional_merge_fields=None):

    """
    merges spectral libraries to a single library
    :param lib_paths: iterable of pathways to libraries
    :param out_path: str
    :param source: iterable, containing a string for each library, to denote source of spectra in new merged library
    :param additional_merge_fields: iterable with strings, additional non-ENVI default metadata fields to merge
    :return: None
    """

    path = lib_paths[0].split('.')[0] + '.hdr'
    metadata = read_envi_header(path)
    wavelengths = metadata['wavelength']
    wavelength_units = metadata['wavelength units']
    fwhm = metadata['fwhm']
    spectra_names_collection = []
    spectra_collection = []
    source_collection = []

    if additional_merge_fields is not None:
        fields = list(additional_merge_fields)
        fields_collection = [[]] * len(fields)
    else:
        fields = None
        fields_collection = None

    for l, lib_path in enumerate(lib_paths):

        spectra, metadata = read_envi_library(lib_path)

        if not np.all(np.equal(wavelengths, metadata['wavelength'])):
            raise ValueError('wavelengths must be equal')
        if wavelength_units != metadata['wavelength units']:
            raise ValueError('wavelength units must be equal')
        if not np.all(np.equal(fwhm, metadata['fwhm'])):
            raise ValueError('FWHM must be equal')

        spectra_collection.append(spectra)
        spectra_names_collection.append(metadata['spectra names'])
        if source is not None:
            source_collection.append([source[l]] * spectra.shape[0])

        if fields:

            for f, field in enumerate(fields):

                fields_collection[f].append(metadata[field])

    spectra = np.concatenate(spectra_collection, axis=0)
    spectra_names = np.concatenate(spectra_names_collection)

    metadata['description'] = 'merged library'
    metadata['spectra names'] = spectra_names
    metadata['lines'] = len(spectra_names)
    if source is not None:
        metadata['source'] = np.concatenate(source_collection)

    if fields:

        for f, field in enumerate(fields):

            metadata[field] = np.concatenate(fields_collection[f])

    save_envi_library(out_path, spectra, metadata)
