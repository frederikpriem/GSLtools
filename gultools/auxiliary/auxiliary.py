import osr
import numpy as np
import copy

from gultools.io import *


def coord2rowcol(x, y, xul, yul, xres, yres):

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

    x = col * xres + xul + xres / 2
    y = yul - row * yres - yres / 2

    return x, y


def transform_coordinates(x_src, y_src, srs_src, srs_dst):

    osr_srs_src = osr.SpatialReference()
    if isinstance(srs_src, str):
        osr_srs_src.ImportFromWkt(srs_src)
    elif isinstance(srs_src, int):
        osr_srs_src.ImportFromEPSG(srs_src)
        
    osr_srs_dst = None
    osr_srs_dst = osr.SpatialReference()
    if isinstance(srs_dst, str):
        osr_srs_dst.ImportFromWkt(srs_dst)
    elif isinstance(srs_dst, int):
        osr_srs_dst.ImportFromEPSG(srs_dst)

    # print(dir(osr_srs_src))
    # print(osr_srs_dst.GetLinearUnitsName())

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


def tile_image(image_path, tile_size, outfolder, basename):

    image, metadata = read_envi_image(image_path, load=False)
    rows = metadata['lines']
    cols = metadata['samples']
    map_info = metadata['map info']
    ulx = map_info[3]
    uly = map_info[4]
    resx = map_info[5]
    resy = map_info[6]
    trows = int(np.ceil(rows / tile_size))
    tcols = int(np.ceil(cols / tile_size))
    ntiles = trows * tcols
    ndigits = len(str(ntiles))

    cnt = -1

    for r in range(trows):

        for c in range(tcols):

            image_tile = image[r:(r + 1) * tile_size, c:(c + 1) * tile_size, :]

            metadata_tile = copy.deepcopy(metadata)
            metadata_tile['lines'] = image_tile.shape[0]
            metadata_tile['samples'] = image_tile.shape[1]
            ulx_tile = ulx + r * tile_size * resx
            uly_tile = uly - c * tile_size * resy
            metadata_tile['map info'][3] = ulx_tile
            metadata_tile['map info'][4] = uly_tile

            cnt += 1
            ndigits_cnt = len(str(cnt))
            name = basename + '_' + (ndigits - ndigits_cnt) * '0' + str(cnt)
            outpath = outfolder + r'\{}.bsq'.format(name)

            save_envi_image(outpath, image_tile, metadata_tile)