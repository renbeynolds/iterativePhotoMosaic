import sys
import os
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import random
import math
import argparse
import operator
import numpy

# Return a cropped and resized version of the image (can upscale)
def cropAndResize(filename, new_width, new_height):
    image = Image.open(filename)
    old_width = image.size[0]
    old_height = image.size[1]
    new_ratio = float(new_width)/new_height
    old_ratio = float(old_width)/old_height
    if old_ratio > new_ratio:
        old_width = int(new_ratio*old_height)
    elif old_ratio < new_ratio:
        old_height = int(old_width/new_ratio)
    image = image.crop((0,0,old_width, old_height))
    image = image.resize((new_width, new_height))
    return image

# Helper function to allow for parellel processing of tile images
def getTile(filename, new_width, new_height):
    image = cropAndResize(filename, new_width, new_height)
    return {'pixels':image.tostring(), 'size':image.size, 'mode':image.mode}

# Return a list of all folders in the given directory
def getFolders(path):
    contained = os.listdir(path)
    subfolders = []
    for item in contained:
        if '.' not in item:
            subfolders.append(item)
    return subfolders

# Return a list of all files with the given extension in the directory
def getFiles(path, extensions):
    contained = os.listdir(path)
    files = []
    for item in contained:
        for extension in extensions:
            if extension in item.lower():
                files.append(item)
    return files

# Return a list of all image files in the directory and all subdirectories
def recursive(path):
    folders = getFolders(path)
    files = [os.path.join(path, f) for f in getFiles(path, ['.jpg'])]
    for folder in folders:
        subpath = os.path.join(path, folder)
        files += recursive(subpath)
    return files

# Calulcate the overall difference between two images
def calculateFitness(image1, image2):
    s = 0
    for band_index, band in enumerate(image1.getbands()):
        m1 = numpy.array([p[band_index] for p in image1.getdata()]).reshape(*image1.size)
        m2 = numpy.array([p[band_index] for p in image2.getdata()]).reshape(*image2.size)
        s += numpy.sum(numpy.abs(m1-m2))
    return s

# Split an image into a grid of cells
def splitImage(image, x_cells, y_cells):
    total_width = image.size[0]
    total_height = image.size[1]
    cell_width = total_width/x_cells
    cell_height = total_height/y_cells
    cells = [x[:] for x in [[0]*y_cells]*x_cells]
    for i in range(0, x_cells):
        for j in range(0, y_cells):
            cell = image.crop((i*cell_width, j*cell_height, (i+1)*cell_width, (j+1)*cell_height))
            cells[i][j] = cell
    return cells

if __name__ == '__main__':

    # Setup and parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', nargs=1, required=True, help='File path for the target image')
    parser.add_argument('--iterations', nargs=1, type=int, default=[1000000], help='Total number of iterations to make')
    parser.add_argument('--intermediate', nargs=1, type=int, help='Will save output image after every n iterations')
    parser.add_argument('--xcells', nargs=1, type=int, default=[64], help='Number of horizontal cells in the output mosaic')
    parser.add_argument('--ycells', nargs=1, type=int, default=[64], help='Number of vertical cells in the output mosaic')
    parser.add_argument('--cellwidth', nargs=1, type=int, default=[30], help='Horizontal pixels per cell')
    parser.add_argument('--cellheight', nargs=1, type=int, default=[18], help='Vertical pixels per cell')
    parser.add_argument('--store', action='store_true', help='Store the processed tiles into a Library')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tiles', nargs=1, help='Directory path containing the tile images')
    group.add_argument('--lib', action='store_true', help='Use preprocessed images in a Library folder')
    args = parser.parse_args()

    print args.iterations
    sys.exit()


    # Constants used for image manipulation
    tile_width = args.cellwidth[0]
    tile_height = args.cellheight[0]
    x_cells = args.xcells[0]
    y_cells = args.ycells[0]
    overall_width = x_cells*tile_width
    overall_height = y_cells*tile_height
    max_uses = 10
    n_cores = multiprocessing.cpu_count()

    # Grab the target image file
    target_image = cropAndResize(args.target[0], overall_width, overall_height)
    target_image_cells = splitImage(target_image, x_cells, y_cells)

    # Grab the tiles image files
    if args.lib:
        tile_files = recursive('Library')
        tile_file = Image.open(tile_files[0])
        if tile_file.size[0] != tile_width or tile_file.size[1] != tile_height:
            print 'Tiles in library are incorrect size'
            sys.exit()
        tile_images = []
        for tile_file in tqdm(tile_files, ncols=50):
            tile_image = Image.open(tile_file)
            tile_image.load()
            tile_images.append({'uses':0, 'image':tile_image})
    else:
        tile_files = recursive(args.tiles[0])
        tile_images = Parallel(n_jobs=n_cores)(delayed(getTile)(tile_file, tile_width, tile_height) for tile_file in tqdm(tile_files, ncols=50))
        tile_images = [{'uses':0, 'image':Image.fromstring(im['mode'], im['size'], im['pixels'])} for im in tile_images]

    if args.store:
        if not os.path.exists('Library'):
            os.makedirs('Library')
        for idx, tile_image in enumerate(tile_images):
            tile_image['image'].save('Library/%d.jpg' %idx)

    # Build the initial mosaic
    mosaic_array = [[0 for x in range(x_cells)] for y in range(y_cells)] 
    for x in tqdm(range(0, x_cells), ncols=50):
        for y in range(0, y_cells):

            choice = random.randint(0, len(tile_images)-1)
            while tile_images[choice]['uses'] == max_uses:
                choice = random.randint(0, len(tile_images)-1)

            tile_images[choice]['uses'] += 1
            cell_fitness = calculateFitness(target_image_cells[x][y], tile_images[choice]['image'])
            mosaic_array[x][y] = {'id':choice, 'fitness':cell_fitness}

    # Start the improvement iterations
    for itr in tqdm(range(0, args.iterations[0]), ncols=50):

        # Generate neighbor 1 and calculate fitness
        x = random.randint(0, x_cells-1)
        y = random.randint(0, y_cells-1)
        choice = random.randint(0, len(tile_images)-1)
        while tile_images[choice]['uses'] == max_uses or mosaic_array[x][y]['id'] == choice:
            choice = random.randint(0, len(tile_images)-1)
        new_fitness1 = calculateFitness(target_image_cells[x][y], tile_images[choice]['image'])
        old_fitness1 = mosaic_array[x][y]['fitness']
        fitness_improvement1 = old_fitness1 - new_fitness1

        # Generate neighbor 2 and calculate fitness
        x1 = random.randint(0, x_cells-1)
        y1 = random.randint(0, y_cells-1)
        x2 = random.randint(0, x_cells-1)
        y2 = random.randint(0, y_cells-1)
        while mosaic_array[x1][y1]['id'] == mosaic_array[x2][y2]['id']:
            x2 = random.randint(0, x_cells-1)
            y2 = random.randint(0, y_cells-1)
        choice1 = mosaic_array[x1][y1]['id']
        choice2 = mosaic_array[x2][y2]['id']
        new_fitness2a = calculateFitness(target_image_cells[x1][y1], tile_images[choice2]['image'])
        new_fitness2b = calculateFitness(target_image_cells[x2][y2], tile_images[choice1]['image'])
        new_fitness2 = new_fitness2a + new_fitness2b
        old_fitness2 = mosaic_array[x1][y1]['fitness'] + mosaic_array[x2][y2]['fitness']
        fitness_improvement2 = old_fitness2  - new_fitness2   

        # Keep the more fit neighbor if it is an improvement
        if fitness_improvement1 > 0 and fitness_improvement2 > 0:
            if fitness_improvement1 > fitness_improvement2:
                # 1
                old_id = mosaic_array[x][y]['id']
                tile_images[old_id]['uses'] -= 1
                tile_images[choice]['uses'] += 1
                mosaic_array[x][y] = {'id':choice, 'fitness':new_fitness1}
            else:
                # 2
                mosaic_array[x1][y1] = {'id':choice2, 'fitness':new_fitness2a}
                mosaic_array[x2][y2] = {'id':choice1, 'fitness':new_fitness2b}
        elif fitness_improvement1 > 0:
            # 1
            old_id = mosaic_array[x][y]['id']
            tile_images[old_id]['uses'] -= 1
            tile_images[choice]['uses'] += 1
            mosaic_array[x][y] = {'id':choice, 'fitness':new_fitness1}
        elif fitness_improvement2 > 0:
            # 2
            mosaic_array[x1][y1] = {'id':choice2, 'fitness':new_fitness2a}
            mosaic_array[x2][y2] = {'id':choice1, 'fitness':new_fitness2b}

        if args.intermediate[0] != None and itr%args.intermediate[0] == 0:
            out_image = Image.new('RGB', (overall_width, overall_height), 'white')
            for x in range(x_cells):
                for y in range(y_cells):
                    out_image.paste(tile_images[mosaic_array[x][y]['id']]['image'], (x*tile_width, y*tile_height))
            out_image.save('%d.jpg' %(itr/args.intermediate[0]))

    out_image = Image.new('RGB', (overall_width, overall_height), 'white')
    for x in range(x_cells):
        for y in range(y_cells):
            out_image.paste(tile_images[mosaic_array[x][y]['id']]['image'], (x*tile_width, y*tile_height))
    out_image.save('mosaic.jpg')
