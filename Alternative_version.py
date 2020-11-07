# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Projet IMED S8
# 
# Ce projet a pour but de vous faire manipuler ET analyser des images médicales : ici des scans de poumons ayant des tumeurs.
#  
# <font color="red"> **Rendu attendu** </font> : ce notebook complété et **bien** commenté. Vous devez faire un notebook *didactique* (comprendre : je ne dois pas me dire "mais qu'est-ce que c'est que ce truc ?" en lisant votre code. Je dois comprendre ce qu'il s'est passé dans vos têtes et juste comprendre votre notebook). Les instructions sont écrites en <font color="blue"> bleu </font>. Les projets sont à me rendre par groupes de 2 ou 3.
#  
# <font color="red"> **Date de rendu** </font> : 30 Août 23h59 (tout retard sera pénalisé). Je vous conseille de me les rendre avant si vous voulez avoir des vacances...
# 
# <font color="blue"> **Instructions**: 
#     * Etape 1 : Segmentez les poumons (cf TP1)
#     * Etape 2 : Segmentez la tumeur
#     * Etape 3 : Exploitez vos résultats : 
#                 ** Quel est le volume de chaque poumon ?
#                 ** Quelle est la taille des tumeurs ?
#     * Critiquez vos résultats : Peut on trouver quel est le poumon gauche ou droit en fonction du volume ? A quel point votre segmentation de tumeur est précise ? 
#     
# <font color="blue"> Vous devez donc faire un programme/une fonction/un bout de code qui, pour une image d'entrée, ressort tout ce qui est demandé dans les "Etapes". Je testerai avec une image que vous n'avez pas : je veux donc en sortie une image avec les poumons segmentés (si je peux me balader dans cette image c'est cool, sinon il faut que je puisse choisir la slice affichée), une image avec la tumeur segmentée (pareil), et les volumes. La partie critique doit apparaître à la fin de votre notebook.</font>
# 
# 
# 
# 
# <font color="red"> **Données** </font> : <font color="black"> Le dossier données comprend les images et les vérités terrains. A chaque image est associée une vérité terrain : cette vérité terrain est là pour vous aider à comprendre où est la tumeur, et pour vous permettre d'évaluer votre segmentation. Attention, quand je testerai votre programme, j'utiliserai des images SANS vérité terrain. Elles ne sont donc là que pour vous permettre de voir où vous en êtes niveau segmentation, ce n'est pas un input supplémentaire.</font> 
# 
# %% [markdown]
# # Nicolas Portal - Clément Rebut - Xavier Fichter
# %% [markdown]
# # Solution proposée
# 
# La solution proposée se décompose en plusieurs grandes étapes:
# * **Pré-traitement:** Débruitage des slices avant de commencer le traitement complet
# * **Segmentation des poumons:** Binariser les slices, segmenter les poumons et calculer leur volume.
# * **Prétraitement avant de segmenter les tumeurs:** traitement des poumons segmentés pour faciliter la segmentation des tumeurs.
# * **Segmentation des tumeurs:** Segmentation des tumeurs et calcul du volume.

# %%
import os
import glob
import numpy as np
import scipy as sp
import nibabel as nib
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## Utilitaires
# %% [markdown]
# Fonction pour afficher les slices

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def display(img, slice_number):
    plt.figure(figsize=(16, 9))
    plt.imshow(img[..., slice_number], cmap='gray')

# %% [markdown]
# Fonction pour charger les données

# %%
def load(path_data):
    img = nib.load(path_data)
    img_data = img.get_fdata()
    img_data = np.rot90(img_data, 1)
    return img_data, img

# %% [markdown]
# ## Pré-traitement
# %% [markdown]
# Fonction appliquant un filtre médian à tous les slices

# %%
from skimage.morphology import binary_closing, area_opening, square, diamond
from skimage.morphology import erosion
from skimage.filters import median

def preprocess(img):
    struct_median = square(5)
    test = img[..., 0]
    first_image = median(test, struct_median)
    for i in range(1, img.shape[2]):
        test = img[..., i]
        test = median(test, struct_median)
        first_image = np.dstack((first_image, test))
    return first_image

# %% [markdown]
# ## Segmentation des poumons
# %% [markdown]
# Cette fonction retourne les slices binarisées avec la methode d'otsu ainsi que ces mêmes slices fermés à l'aide d'un element structurant sous la forme d'un disque de diametre égale à 6

# %%
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, square, diamond, disk

def image_segmentation_otsu(img):
    struct_closing = disk(6)
    test = img[..., 0]
    thresh = threshold_otsu(test)
    first_image = test > thresh
    first_image = np.invert(first_image)
    first_image_closed = (binary_closing(first_image, struct_closing)).astype(np.uint8) * 255
    for i in range(1, img.shape[2]):
        test = img[..., i]
        thresh = threshold_otsu(test)
        binary = test > thresh
        binary = np.invert(binary)
        binary_closed = (binary_closing(binary, struct_closing)).astype(np.uint8) * 255
        first_image_closed = np.dstack((first_image_closed, binary_closed))
        first_image = np.dstack((first_image, binary))
    return first_image_closed, first_image

# %% [markdown]
# Cette fonction permet de retirer les objets qui touchent le bord de l'image. Cela permet de ne garder que les poumons segmentés.
# 
# Afin de retirer les objets indésirables, une reconstruction est utilisée en utilisant une graine qui fait le tour de l'image. Les objets obtenus avec la recustruction sont retirés de l'image et seuls les poumons restent.

# %%
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction
import random

def get_segmented_lung(otsu, otsu_closed):
    out_image = otsu.copy()
    for i in range(otsu.shape[2]):
        rec = np.ones(shape=(otsu.shape[0], otsu.shape[1]))
        rec[1:rec.shape[0]-1, 1:rec.shape[1]-1] = 0
        seed = rec * out_image[..., i]
        border = reconstruction(seed, otsu[..., i]).astype(np.uint8)
        out_image[..., i] = out_image[..., i] - border
    return out_image

# %% [markdown]
# Cette fonction efface les formes qui se trouvent au dessus du point le plus haut des poumons ou au dessous du point le plus bas des poumons ou a gauche du point le plus à gauche des poumons ou a droite du point le plus à droite des poumons. Cette fonction elimine également les formes se trouvant entre les deux poumons.

# %%
from skimage.measure import points_in_poly
from skimage.segmentation import flood, flood_fill
from scipy.ndimage.morphology import binary_fill_holes

def remove_blobs(lung):
    col = lung.shape[1]
    out_image = np.copy(lung)
    for i in range(lung.shape[2]):

        if np.count_nonzero(lung[..., i]) == 0:
            continue

        filled = (binary_fill_holes(lung[..., i])).astype(np.uint8) * 255
        label_image, num = label(filled, return_num=True)

        if num == 1:
            continue

        max_region_area = 0
        max_region = 0
        second_max_region_area = 0
        second_max_region = 0
        j = 0
        regions = regionprops(label_image)
        for index, region in enumerate(regions):
            if region.area > max_region_area:
                max_region_area = region.area
                max_region = region.coords
                j = index

        del regions[j]

        for region in regions:
            if region.area > second_max_region_area:
                second_max_region_area = region.area
                second_max_region = region.coords

        left = min(np.min(max_region[:, 1]), np.min(second_max_region[:, 1]))
        right = max(np.max(max_region[:, 1]), np.max(second_max_region[:, 1]))
        top = min(np.min(max_region[:, 0]), np.min(second_max_region[:, 0]))
        bottom = max(np.max(max_region[:, 0]), np.max(second_max_region[:, 0]))

        for region in regionprops(label_image):
            if ((region.centroid[1] > (col // 2) - (col // 8) and region.centroid[1] < (col // 2) + (col // 8) and region.area < 1500) or                         region.centroid[0] < top or region.centroid[0] > bottom or region.centroid[1] > right or region.centroid[1] < left) :
                centroid = tuple(int(x) for x in region.centroid)
                label_image = flood_fill(label_image, centroid, 0)
        out_image[..., i][label_image == 0] = 0
    return out_image

# %% [markdown]
# Cette fonction renvoit les poumons correctement segmentés ainsi que le nombre de pixels dans le poumon gauche et droit de chaque slices. On ferme les poumons et bouche les trous.
# 
# Si les deux poumons ne font qu'un, on les détache en appliquant une ouverture dont l'élément structurant est de taille croissante. Le nombre de pixels renvoyé correspond à la somme de tous les pixels du poumon gauche suivant toutes les slices, ainsi que la somme de tous les pixels du poumon droit suivant toutes les slices. Cela va permettre de calculer le volume de chaque poumon.

# %%
from skimage.morphology import rectangle

def compute_volume_pixels(lung):
    col = lung.shape[1]
    left_volume = 0
    right_volume = 0
    struct_opening_post = disk(2)
    struct_closing = disk(10)
    filled = np.copy(lung)
    label_image = np.zeros(lung[..., 0].shape)
    max_area = 0
    for i in range(lung.shape[2]):

        if np.count_nonzero(lung[..., i]) == 0:
            continue

        label_image_first, num = label(lung[..., i], return_num=True)
        coords = regionprops(label_image_first)[0].coords
        threshold = coords[0, 0] + ((coords[-1, 0] - coords[0, 0]) // 8)

        filled_test = (binary_fill_holes(lung[..., i])).astype(np.uint8) * 255
        closed_test = filled_test.copy()
        closed_test = (binary_closing(filled_test, struct_closing)).astype(np.uint8) * 255
        closed_test = (binary_opening(closed_test, struct_opening_post)).astype(np.uint8) * 255

        label_image, num = label(closed_test, return_num=True)

        size = 5
        opened_high = closed_test.copy()
        while size <= 40 and num == 1 and regionprops(label_image)[0].area > 10000:
            struct_opening = disk(size)
            opened_high = (binary_opening(opened_high, struct_opening)).astype(np.uint8) * 255
            label_image, num = label(opened_high, return_num=True)
            size += 5
        if size <= 40:
            closed_test = opened_high

        filled_test_second = (binary_fill_holes(closed_test)).astype(np.uint8) * 255
        #filled_test_second = (binary_opening(filled_test_second, struct_opening_post)).astype(np.uint8) * 255

        label_image, num = label(filled_test_second, return_num=True)
    
        filled[..., i] = filled_test_second
        
        area = 0
        for region in regionprops(label_image):
            area += region.area
            if np.sum(region.coords[:, 1] > col // 2) <= np.sum(region.coords[:, 1] <= col // 2):
                left_volume += region.area
            else:
                right_volume += region.area

    return left_volume, right_volume, filled

# %% [markdown]
# ## Pré-traitement pour segmenter les tumeurs 
# %% [markdown]
# Cette fonction permet de remplir les crevasses dans les poumons. Cela va permettre d'identifier les tumeurs au bord des poumons. Pour cela on utilise le watershed sur l'inverse de l'image. Puis grace à une approximation polynomiale de la convex hull, on determine si la zone doit être remplie.
# 
# Si tous les pixels de la zone ne sont pas dans la convex hull et si le centroïde de la zone n'est pas dans la convex hull alors on ne rempli pas la zone.

# %%
from skimage.morphology import convex_hull_object
from skimage.measure import find_contours
from skimage.measure import approximate_polygon
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def fill(lung):
    struct = disk(5)
    test = lung.copy()
    for i in range(lung.shape[2]):
        test[..., i] = binary_fill_holes(lung[..., i])

        label_test, num = label(test[..., i], return_num=True)
        if num != 2:
            continue

        chull = convex_hull_object(test[..., i])
        coords = []
        for contour in find_contours(chull, 0):
            coords.append(approximate_polygon(contour, tolerance=0))
        
        if len(coords) < 2:
            continue

        inverted = np.invert(test[..., i])
        distance = ndi.distance_transform_edt(inverted)
        peaks = peak_local_max(distance, labels=inverted)

        left_peaks = points_in_poly(peaks, coords[0])
        right_peaks = points_in_poly(peaks, coords[1])
        peaks_mask = left_peaks | right_peaks
        peaks = peaks[peaks_mask]
        peak_image = np.zeros(lung[..., i].shape)
        peak_image[peaks[:, 0], peaks[:, 1]] = 1

        if len(peaks == 1):
            peak_image[0, 0] = 1

        markers = ndi.label(peak_image)[0]
        labels = watershed(-distance, markers, mask=inverted)
        labels[labels == 1] = 0

        for region in regionprops(labels):
            centroid = np.asarray([int(x) for x in region.centroid]).reshape(1, 2)
            if ((np.sum(points_in_poly(region.coords, coords[0])) < region.area and
            np.sum(points_in_poly(region.coords, coords[1])) < region.area) or 
            (not points_in_poly(centroid, coords[1]) and not points_in_poly(centroid, coords[1]))):
                centroid = tuple(int(x) for x in region.centroid)
                labels = flood_fill(labels, centroid, 0)

        labels[labels > 0] = 255
        test[..., i][labels > 0] = 255

    return test

# %% [markdown]
# Cette fonction fait la différence entre les poumons pleins et les poumons non pleins. Cela permet de récupérer les régions pouvant être des tumeurs. On utilise le watershed basé sur la distance au fond pour effacer les régions dont moins de la moitié des pixels se trouvent dans la zone des poumons.

# %%
def get_sub(lung, segmented_lung):
    struct_closing = disk(20)
    struct_opening = disk(3)
    sub = lung.copy()
    for i in range(lung.shape[2]):

        if np.count_nonzero(lung[..., i]) == 0:
            sub[..., i] = np.zeros(lung[..., i].shape)
            continue

        test = lung[..., i]

        coords = []
        for contour in find_contours(segmented_lung[..., i], 0):
            coords.append(approximate_polygon(contour, tolerance=0))

        sub[..., i] = test.astype(np.uint8) - segmented_lung[..., i].astype(np.uint8)

        distance = ndi.distance_transform_edt(sub[..., i])
        local_maxi = peak_local_max(distance, indices=False, labels=sub[..., i])
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=sub[..., i])

        for region in regionprops(labels):
            remove  = True
            for coord in coords:
                if np.sum(points_in_poly(region.coords, coord)) > (region.area // 2):
                    remove = False
            if remove:
                centroid = tuple(int(x) for x in region.centroid)
                labels = flood_fill(labels, centroid, 0)

        labels[labels > 0] = 255
        sub[..., i] = labels

    return sub

# %% [markdown]
# ## Segmentation des tumeurs
# %% [markdown]
# ## Segmentation préliminaire
# %% [markdown]
# Cette fonction effectue une première segmentation des tumeurs. On selectionne la zone de plus grande taille. On tient compte de l'eccentricité de la region ainsi que du rapport du nombre de pixels dans la region sur le nombre de pixels de la bounding box (extent). On selectionne également uniquement les régions dont la taille n'est pas trop grande ou trop petite.

# %%
def get_tumors(sub):
    tumors = np.zeros(sub.shape)
    struct = disk(4)
    for i in range(sub.shape[2]):
        if np.count_nonzero(sub[..., i]) == 0:
            continue
        
        test = binary_opening(sub[..., i], struct)
        label_image = label(test)

        list_of_data = []
        for region in regionprops(label_image):
            data = []
            centroid = tuple(int(x) for x in region.centroid)
            if region.eccentricity < 0.9 and region.extent > 0.5 and region.area < 3600 and region.area > 50:
                data.append(centroid)
                data.append(region.area)
                list_of_data.append(data)
        
        if not list_of_data:
            continue

        centroid = max(list_of_data, key=lambda item:item[1])[0]
        segmented_tumor = flood_fill(label_image, centroid, 255)
        segmented_tumor[segmented_tumor < 255] = 0
        tumors[..., i] = segmented_tumor
    tumors /= 255
    return tumors

# %% [markdown]
# ## Assemblage des slices de tumeurs
# %% [markdown]
# Cette fonction permet de récupérer les séquences de slices où l'on a trouvé une tumeur dans la même region. Ces séquences sont sous la forme de sous-listes au sein d'une liste. Ces sous-listes sont composées des numéros des slices. Pour se faire, on récupère les coordonnés de l'aproximation polynomiale de chaque tumeur pour chaque slice. Ensuite on compare le centroid de la tumeur dans la slice courante aux coordonnés du polynome dans la slice précédente. Si le centroid se situe à l'intérieur du polynome alors il s'agit de la même région et donc de la même sous-liste.

# %%
def get_sequences(tumors):
    list_of_sequence = []
    sequence = []
    j = 0
    while np.count_nonzero(tumors[..., j]) == 0:
        j += 1
    sequence.append(j)
    label_image = label(tumors[..., j])
    contour = find_contours(label_image, 0)[0]
    coords_region_previous = approximate_polygon(contour, tolerance=0)
    previous_index = j
    for i in range(j + 1, tumors.shape[2]):
        if np.count_nonzero(tumors[..., i]) == 0:
            continue
        label_image = label(tumors[..., i])
        centroid = regionprops(label_image)[0].centroid
        centroid = np.asarray([int(x) for x in centroid]).reshape(1, 2)
        if points_in_poly(centroid, coords_region_previous) and abs(i - previous_index) < 3:
            sequence.append(i)
        else:
            list_of_sequence.append(sequence)
            sequence = []
            sequence.append(i)
        contour = find_contours(label_image, 0)[0]
        coords_region_previous = approximate_polygon(contour, tolerance=0)
        previous_index = i
    list_of_sequence.append(sequence)
    return list_of_sequence

# %% [markdown]
# Cette fonction permet de rassembler les sous-listes séparées par une sous-liste de taille 1. En effet, il est possible qu'au sein d'une suite de tumeurs dans la même région, une tumeur dans une slice ait été incorrectement ségmentée. On itére sonc ici sur 3 sous-listes à la fois. On s'assure que la region de tumeur de la dernière slice de la séquence précedente et de la première slice de la séquence suivante soient similaires.

# %%
def merge_sublist(list_of_sequence, tumors):
    flag = True
    while flag:
        merged_sublists = []
        for i in range(1, len(list_of_sequence) - 2):
            if (len(list_of_sequence[i]) == 1 and len(list_of_sequence[i - 1]) >= 2 and len(list_of_sequence[i + 1]) >= 2 and
            abs(list_of_sequence[i - 1][-1] - list_of_sequence[i][0]) < 3 and abs(list_of_sequence[i][-1] - list_of_sequence[i + 1][0]) < 3):

                label_image = label(tumors[..., list_of_sequence[i - 1][-1]])
                contour = find_contours(label_image, 0)[0]
                coords_region_previous = approximate_polygon(contour, tolerance=0)

                label_image = label(tumors[..., list_of_sequence[i + 1][0]])
                centroid = regionprops(label_image)[0].centroid
                centroid = np.asarray([int(x) for x in centroid]).reshape(1, 2)

                if points_in_poly(centroid, coords_region_previous):
                    new_sublist = []
                    new_sublist = list_of_sequence[i - 1] + list_of_sequence[i] + list_of_sequence[i + 1]
                    merged_sublists += (new_sublist, i - 1)
                    flag = True
                    break
            flag = False
        if flag:
            del list_of_sequence[merged_sublists[1]]
            del list_of_sequence[merged_sublists[1]]
            del list_of_sequence[merged_sublists[1]]
            list_of_sequence.append(merged_sublists[0])
            list_of_sequence.sort(key=lambda x:x[0])
    return list_of_sequence

# %% [markdown]
# Cette fonction permet de récupérer les numéros de slices compris dans des sous-listes de taille supérieur à un seuil. Ce seuil est calculé en se basant sur la taille moyenne des séquences de taille supérieur à 2.

# %%
def filter_sublists(list_of_sequence):
    list_above_two = [x for x in list_of_sequence if len(x) >= 2]
    threshold = round(sum([len(x) for x in list_above_two]) / len(list_above_two))
    list_of_sequence = [x for x in list_of_sequence if len(x) >= threshold]
    list_of_sequence = [j for i in list_of_sequence for j in i]
    return list_of_sequence

# %% [markdown]
# Cette fonction permet de récupérer la region la plus probable de la tumeur. Pour cela on fait la somme de tous les pixels de chaque slice obtenus dans la fonction précédente. On obtient ainsi une carte des probabilités. Ensuite, pour chaque région de la carte de probabilité, on obtient l'intensité moyenne de la région (grace à la fonction regionprops).
# 
# La région ayant l'intensité moyenne la plus élevée au sein de l'image (carte de probabilité) est désignée comme étant la région de recherche.
# 
# Avant cela on a pris soin de ramener les valeurs de la carte de probabilité entre 0 et 255. Les régions de la carte des probabilités ne satisfaisant pas certains critères sont éliminées.

# %%
def get_tumor_region(tumors, list_of_sequence):
    struct = disk(2)

    res = np.sum(tumors[..., list_of_sequence], axis=2)

    min_value = np.min(res)
    max_value = np.max(res)

    res = (((res - min_value) / (max_value - min_value)) * 255).astype(np.uint8)

    thresh = threshold_otsu(res)
    res_otsu = res > thresh

    #res_otsu = binary_opening(res_otsu, struct)

    label_image = label(res_otsu)
    regions = regionprops(label_image)

    for region in regions:
        if region.eccentricity > 0.9 or region.extent < 0.6 or region.area > 3000 or region.area < 150:
            centroid = tuple(int(x) for x in region.centroid)
            label_image = flood_fill(label_image, centroid, 0)

    label_image[label_image > 0] = 255
    res[label_image == 0] = 0

    l, num = label(label_image, return_num=True)
    if num < 1:
        label_image = label(res_otsu)
        regions = regionprops(label_image)

        for region in regions:
            if region.eccentricity > 0.9 or region.extent < 0.5 or region.area > 3000 or region.area < 150:
                centroid = tuple(int(x) for x in region.centroid)
                label_image = flood_fill(label_image, centroid, 0)

        label_image[label_image > 0] = 255
        res[label_image == 0] = 0

        l, num = label(label_image, return_num=True)

    regions = regionprops(l, intensity_image=res)
    mean_intensity = [x.mean_intensity for x in regions]
    max_intensity = np.max(mean_intensity)
    for index, region in enumerate(regions):
        if region.mean_intensity < max_intensity:
            centroid = tuple(int(x) for x in region.centroid)
            label_image = flood_fill(label_image, centroid, 0)

    area = regionprops(label_image)[0].area

    contour = find_contours(label_image, 0)[0]
    coords_region = approximate_polygon(contour, tolerance=0)

    return coords_region, area, label_image

# %% [markdown]
# Cette fonction permet de récupérer le numéro de slice inférieur et le numéro de slice supérieur (intervalle) de la région de recherche obtenue dans la fonction precedente. Il s'agit donc du numéro de slice inférieur et supérieur des slices ayant permis d'obtenir la région de plus forte probabilité.
# 
# Pour se faire, on compare les coordonnés de l'aproximation polynomiale de la region de recherche avec les coordonnés des zones tumorales de chaque slice obtenu grâce à la fonction "filter_sublists".

# %%
def get_search_area(list_of_sequence_merged, tumors, area, coords_region):
    list_above_two = [x for x in list_of_sequence_merged if len(x) >= 2]
    list_of_ratio = []
    for sublist in list_above_two:
        res = np.sum(tumors[..., sublist], axis=2)
        min_value = np.min(res)
        max_value = np.max(res)

        res = (((res - min_value) / (max_value - min_value)) * 255).astype(np.uint8)

        thresh = threshold_otsu(res)
        res_otsu = res > thresh

        l, num = label(res_otsu, return_num=True)

        coords = regionprops(l)[0].coords

        good = np.sum(points_in_poly(coords, coords_region))

        ratio = good / area

        list_of_ratio.append((ratio, sublist[0], sublist[-1]))
        max_value = max(list_of_ratio, key=lambda item:item[0])
    return max_value[1], max_value[2]

# %% [markdown]
# ## Segmentation finale des tumeurs
# %% [markdown]
# Cette fonction effectue la deuxième et dernière ségmentation des tumeurs. Elle est presque identique à la fonction effectuant la première segmentation. Elle tient compte cependant de la region de plus forte probabilité calculée precedemment. C'est a dire qu'on accepte uniquement les zones potentielles dont au moins la moitié des pixels sont présent dans la zone la plus probable (calculée précédemment).

# %%
def get_tumors_post(sub, coords, min_slice, max_slice):
    struct = disk(4)
    tumors = np.zeros(sub.shape)
    for i in range(min_slice - 10, max_slice + 10):
        if np.count_nonzero(sub[..., i]) == 0:
            continue

        test = binary_opening(sub[..., i], struct)
        label_image = label(test)

        list_of_data = []
        for region in regionprops(label_image):
            data = []
            centroid = tuple(int(x) for x in region.centroid)
            if (region.eccentricity < 0.95 and region.extent > 0.45 and region.area < 3600 and region.area > 50
            and np.sum(points_in_poly(region.coords, coords)) > region.area // 2):
                data.append(centroid)
                data.append(region.area)
                list_of_data.append(data)
        
        if not list_of_data:
            continue

        centroid = max(list_of_data, key=lambda item:item[1])[0]
        segmented_tumor = flood_fill(label_image, centroid, 255)
        segmented_tumor[segmented_tumor < 255] = 0
        tumors[..., i] = segmented_tumor
    tumors /= 255
    return tumors

# %% [markdown]
# Fonction qui calcule le dice score de la segmentation au niveau des pixels.

# %%
def dice_score(truth, predictions):
    dice = np.sum(predictions[truth])*2.0 / (np.sum(predictions) + np.sum(truth))
    return dice

# %% [markdown]
# Cette fonction calcule le volume de chaque poumon. On utilise la taille des voxels contenu dans le header du fichier nii pour cela.

# %%
def get_volume(left_pixels, right_pixels, img):
    header = img.header
    zoom = header.get_zooms()
    volume_voxel = zoom[0] * zoom[1] * zoom[2]
    volume_left = volume_voxel * right_pixels
    volume_right = volume_voxel * left_pixels
    return volume_left / 1000000, volume_right / 1000000

# %% [markdown]
# Cette fonction calcule le volume de la tumeur. On fait la somme de l'image 3d des tumeurs ségmentées.

# %%
def get_volume_tumor(first_tumor, img):
    pixels = np.sum(first_tumor / 255)
    header = img.header
    zoom = header.get_zooms()
    volume_voxel = zoom[0] * zoom[1] * zoom[2]
    volume = volume_voxel * pixels
    return volume / 1000000

# %% [markdown]
# # Pipeline complète
# %% [markdown]
# Fonction permettant de traiter les images. Cette fonction execute toutes les fonctions décrites précédemment. Elle renvoit:
# 
# 1. les poumons ségmentés
# 2. les tumeurs ségmentés
# 3. Le volume du poumon gauche (en Litre)
# 4. Le volume du poumon droit (en Litre)
# 5. Le volume de la tumeur (en Litre)
# 
# Elle permet également de visualiser les résultats et de se déplacer dans les slices.

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def get_data(path_data):
    img_data, img = load(path_data)
    preprocessed_img = preprocess(img_data)
    otsu_closed, otsu = image_segmentation_otsu(preprocessed_img)
    lung = get_segmented_lung(otsu, otsu_closed)
    lung_removed = remove_blobs(lung)
    left_pixels, right_pixels, filled_lung = compute_volume_pixels(lung_removed)
    filled_lung = remove_blobs(filled_lung)
    segmented_lung = fill(filled_lung)
    segmented_lung = remove_blobs(segmented_lung)
    sub = get_sub(lung_removed, segmented_lung)
    tumors = get_tumors(sub)
    list_of_sequence = get_sequences(tumors)
    list_of_sequence_merged = merge_sublist(list_of_sequence, tumors)
    list_of_sequence_filtered = filter_sublists(list_of_sequence_merged)
    coords_region, area, image_region = get_tumor_region(tumors, list_of_sequence_filtered)
    min_slice, max_slice = get_search_area(list_of_sequence_merged, tumors, area, coords_region)
    tumors_post = get_tumors_post(sub, coords_region, min_slice, max_slice)
    volume = get_volume(left_pixels, right_pixels, img)
    volume_tumor = get_volume_tumor(tumors_post, img)
    return segmented_lung, tumors_post, volume[0], volume[1], volume_tumor

lung_segmented, tumor_segmented, volume_gauche, volume_droit, volume_tumor = get_data(???)

print("volume poumon gauche: {}".format(volume_gauche))
print("volume poumon droit: {}".format(volume_droit))
print("volume tumeur: {}".format(volume_tumor))

interact(display, img=fixed(lung_segmented), slice_number=widgets.IntSlider(min=0, max=lung_segmented.shape[2] - 1, step=1, value=0))
interact(display, img=fixed(tumor_segmented), slice_number=widgets.IntSlider(min=0, max=lung_segmented.shape[2] - 1, step=1, value=0))

# %% [markdown]
# # Résultats et commentaires 
# %% [markdown]
# La méthode utilisée ici pour ségmenter les tumeurs est loin d'être optimale. Elle donne des résultats satisfaisant pour les cas 3, 9 et 10. La principale difficulté réside dans l'ajustement des paramètres concernant les zones de tumeur potentielle. Il est crucial d'identifier la bonne région de plus forte probabilité pour la tumeur sinon on obtient un dice score de 0. Dans le cas ou cette région de probabilité maximum est correctement identifiée, les résultats sont alors très bon car la recherche ne s'effectue que dans cette zone. Pour identifier correctement cette région, la première phase de segmentation (fonction "get_tumors") joue un rôle primordial. En ajustant les paramêtres (extent, taille min ,taille max, exentricité), il faut être capable d'identifier les tumeurs dans les slices présentant réellement une tumeur.
# 
# La principale difficulté réside dans la grande heterogeneité des données. En utilisant certains parametre, il est possible d'obtennir de bons résultats pour un cas; mais ces paramètres ne seront plus adaptés pour un autre cas. Plus précisemment, l'extent (rapport du nombre de pixels dans l'air sur le nombre de pixels dans la bounding box) joue un rôle crucial dans notre methode pour filtrer les zones de tumeur potentielles.
# 
# Il est possible d'identifier le poumon droit du poumon gauche en se basant sur le volume de ces derniers. On constate en effet que le poumon droit a toujours un volume legerement plus important que le volume du poumon gauche.
# 
# Cependant, dans certains cas on trouve un volume du poumon gauche supérieur au volume du poumon droit. Cela est du à la fonction "fill" qui rempli les crevasses dans les poumons. Si le poumon gauche présente plus de crevasses que le poumon droit, alors il gagnera un nombre significatif de pixels quand ces crevasses seront bouchées. Cependant, nous n'avons pas trouvé d'alternative à cette fonction "fill" pour correctement identifier les tumeurs se trouvant au bord des poumons.
# 
# Les volumes sont en litres.
# 
# cas 1:
# 
#     dice_score: 0.0
# 
#     volume poumon droit: 2.1398890583992003
# 
#     volume poumon gauche: 1.727829556465149
#     
#     volume tumeur:
# 
# cas 3:
# 
#     dice_score: 0.0
# 
#     volume poumon droit: 3.540152400970459
# 
#     volume poumon gauche: 3.456832363128662
#     
#     volume tumeur:
# 
# cas 4:
# 
#     dice_score: 0.0
# 
#     volume poumon droit: 3.261168833074272
# 
#     volume poumon gauche: 2.8309207727301122
# 
#     volume tumeur:
# 
# cas 5:
# 
#     dice_score: 0.9175999322971029
# 
#     volume poumon droit: 2.118578227806091
# 
#     volume poumon gauche: 1.7583750380516052
# 
#     volume tumeur: 3.160249897077972e-05
# 
# cas 6:
# 
#     dice_score: 0.7374476987447699
# 
#     volume poumon droit: 2.6268479781603813
# 
#     volume poumon gauche: 2.7642347588413956
# 
#     volume tumeur: 1.0336136299021098e-05
# 
# cas 9:
# 
#     dice_score: 0.7661260999200058
# 
#     volume poumon droit: 2.9753604266204836
# 
#     volume poumon gauche: 3.0345497516021727
# 
#     volume tumeur: 0.00028414032079659263
#     
# cas 10:
# 
#     dice_score: 0.8525568799541402
# 
#     volume poumon droit: 1.8121814800071716
# 
#     volume poumon gauche: 1.3545287705751659
#     
#     volume tumeur: 4.0758489387175626e-05
# 
# Un des problèmes majeures vient du fait qu'en bouchant les vaisseaux des poumons et les crevasses (fonction "fill") au moment de la segmentation, puis en faisant la soustraction entre l'image non bouchée et l'image bouchée, on se retrouve avec des vaisseaux ou des crevasses qui sont parfois analysés comme étant des tumeurs. Cela contribue à nos résultats sous optimaux. D'un autre côté on ne peux pas se passer de la fonction "fill" pour remplir les zones au bords des poumons où l'on peut trouver des tumeurs (cas 10 et 9 par example).

