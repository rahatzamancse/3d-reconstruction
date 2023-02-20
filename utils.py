import numpy as np
import plotly.graph_objs as go
import cv2
from tqdm import tqdm
import itertools
# import open3d as o3d
import io
import os
from PIL import Image
import tensorflow as tf
import copy
import pandas as pd
from pyoints import (
    storage,
    Extent,
    transformation,
    filters,
    registration,
    normals,
)
import trimesh
from functools import cache
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import MDS
import json

shapenet_category_to_id = {
    'airplane'	: '02691156',
    'bench'		: '02828884',
    'cabinet'	: '02933112',
    'car'		: '02958343',
    'chair'		: '03001627',
    'lamp'		: '03636649',
    'monitor'	: '03211117',
    'rifle'		: '04090263',
    'sofa'		: '04256520',
    'speaker'	: '03691459',
    'table'		: '04379243',
    'telephone'	: '04401088',
    'vessel'	: '04530566'
}

def get_baseline_metrics(points, dist_mats):
    mds = MDS(n_components=3, dissimilarity='precomputed')
    embeddings = mds.fit_transform(np.array(dist_mats).mean(axis=0))
    trans_mat, loss = get_4pointsample_transform_mat(embeddings, points)
    return {
        'chamfer': chamfer_distance_numpy(np.expand_dims(apply_transformation(embeddings, trans_mat), axis=0), np.expand_dims(points, axis=0)),
        'EMD': emd(apply_transformation(embeddings, trans_mat), points),
        'trans_mat': trans_mat,
        'embeddings': embeddings
    }

shapenet_id_to_category = dict(reversed(list(shapenet_category_to_id.items())))

def emd(X, Y, scale=100):
    d = cdist(X, Y)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum()*scale / min(len(X), len(Y))

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

@cache
def load_modelnet():
    return tf.keras.utils.get_file(
        "modelnet.zip",
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        extract=True,
    )

def get_shapenet_mesh(dataset, datadir='', mesh=False):
    _, model_id, obj_id = dataset.split(':')
    model_id = shapenet_category_to_id[model_id]
    obj_path = os.path.join(datadir, model_id, obj_id, 'models/model_normalized.obj')
    return as_mesh(trimesh.load(obj_path)) if mesh else trimesh.load(obj_path)
    

def get_dataset_points(dataset, datadir, n_points=1024, normalize=False):
    if dataset == 'toy_points':
        points = pd.read_csv(f'{datadir}/fullData.csv', header=None).T.to_numpy()
        
    elif dataset.split(':')[0] == "ModelNet10":
        datadir = load_modelnet()
        _, object_name, object_idx = dataset.split(':')
        obj_path = os.path.join('/'.join(datadir.split('/')[:-1]), f"ModelNet10/{object_name}/train/{object_name}_{object_idx}.off")
        print(obj_path)
        mesh = trimesh.load(obj_path)
        points = mesh.sample(n_points)
        
    elif dataset.split(':')[0] == "ShapeNet":
        _, object_name, object_idx = dataset.split(':')
        object_name = shapenet_category_to_id[object_name]
        scene_or_mesh = trimesh.load(os.path.join(datadir, f"{object_name}/{object_idx}/models/model_normalized.obj"))
        points = as_mesh(scene_or_mesh).sample(n_points)
        
    elif dataset.split(':')[0] == 'Pix3D':
        with open(os.path.join(datadir, 'pix3d.json')) as f:
            metadata = json.load(f)
        _, category, obj_id = dataset.split(':')
        img_names = [f'img/{category}/{obj_id}.png', f'img/{category}/{obj_id}.jpg']
        obj_path = None
        for obj in metadata:
            if 'img' in obj and obj['img'] in img_names:
                obj_path = obj['model']
                break
        scene_or_mesh = trimesh.load(os.path.join(datadir, obj_path))
        points = as_mesh(scene_or_mesh).sample(n_points)
        
    if normalize:
        for i in range(3):
            points[:, i] = (points[:, i] - points[:, i].min()) / (points[:, i].max() - points[:, i].min())
        
    return points

# START: Reichstag dataset
def get_keypoints(imgs, feature_point_detector = None):
    if not feature_point_detector:
        feature_point_detector = cv2.SIFT_create()
    all_keypoints, all_descriptors = [], []
    for img in tqdm(imgs, total=len(imgs), desc="Getting Feature points"):
        keypoints, descriptors = feature_point_detector.detectAndCompute(img, None)
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)
    return all_keypoints, all_descriptors

def match_keypoints(descriptor1, descriptor2, matcher=None, dist_thresh=2000):
    if not matcher:
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # sort matches by distance and filter by threshold
    matches = sorted(
        filter(
            lambda x: x.distance < dist_thresh, 
            matcher.match(
                descriptor1, 
                descriptor2
            )
        ), 
        key = lambda x:x.distance
    )
    return matches
def matches_summary(matches, N=5, img1=1, img2=2):
    print(f'''Total matches: {len(matches)}
    First and last 5 query index of Image# {img1}: {list(match.queryIdx for match in matches[:N])} & {list(match.queryIdx for match in matches[-N:])}
    First and last 5 train index of Image# {img2}: {list(match.trainIdx for match in matches[:N])} & {list(match.trainIdx for match in matches[-N:])}
    ''')
# END: Reichstag dataset

# START: Alignment and CV Operations
def getRotationMatrix(angle, axis):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)

    if axis == "x":
        return np.array([
            [1, 0, 0], 
            [0, c, -s],
            [0, s, c]
        ])
    if axis == "y":
        return np.array([
            [c, 0, s], 
            [0, 1, 0],
            [-s, 0, c]
        ])
    if axis == "z":
        return np.array([
            [c, -s, 0], 
            [s, c, 0],
            [0, 0, 1]
        ])

    
def plot_3D_paper(samples_3D, range_x=None, range_y=None, range_z=None, proj_type='perspective', colors=None, eq_range=0, cubic=True, pad=0.4, point_size=2, opacity=1):
    '''\
    Plots sets of 3D points.
    ----------
    samples_3D : List[ndarray], each of shape (n_samples, 3)
        List of Set of points.
    range_x, range_y, range_z : List[int] -> [min, max] 
        The range of the plot.
    proj_type: str
        Type of projection. ('orthographic' or 'perspective')
    Returns
    -------
    fig : go.Figure
        The plotly Figure object
    '''
    plots = []
    if not colors:
        colors = ['green', 'red', 'blue', 'black'][:len(samples_3D)]
        while len(colors) != len(samples_3D):
            colors.append('red')
    assert len(colors) == len(samples_3D), "Color must match the number of samples_3D"
    for sample_3D, color in zip(samples_3D, colors):
        plots.append(go.Scatter3d(
            x=sample_3D[:, 0],
            y=sample_3D[:, 1],
            z=sample_3D[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                opacity=opacity,
                color=sample_3D[:, 2] if color == 'fancy' else color,
                colorscale='hsv',
                symbol='circle',
            ),
            legendgroup=color,
            showlegend=True,
            name=color
        ))

    layout = go.Layout(
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
        ), 
        # autosize=True,
        scene=dict(
            xaxis_visible=False, yaxis_visible=False,zaxis_visible=False
        )
    )
    
    fig = go.Figure(data=[*plots], layout=layout)
    return fig

    
def get_equiangle_persps(points, n):
    if n == 1:
        return [points], np.identity(3)
    perspectives = []
    projection_mats = []
    if n % 2 == 1:
        perspectives.append(points)
        projection_mats.append(np.identity(3))
        n -= 1
    
    add_angle = 360/n
    for i in range(1, n//2+1):
        angle = i*add_angle
        rot_mat = getRotationMatrix(angle, 'y')
        rot_mat2 = getRotationMatrix(-angle, 'y')
        perspectives.append(np.dot(points, rot_mat))
        projection_mats.append(rot_mat)
        perspectives.append(np.dot(points, rot_mat2))
        projection_mats.append(rot_mat2)
    return perspectives, projection_mats


def get_randomized_all_persps(points, n):
    if n == 1:
        flattened_points = points.copy()
        flattened_points[:, 2] = 0
        return [flattened_points], np.identity(3)
    perspectives = []
    projection_mats = []
    if n % 2 == 1:
        flattened_points = points.copy()
        flattened_points[:, 2] = 0
        perspectives.append(flattened_points)
        projection_mats.append(np.identity(3))
        n -= 1
    
    add_angle = 360/n
    for i in range(1, n//2+1):
        angle_range = [(i-1)*add_angle, i*add_angle]
        angle1 = np.random.uniform()*(angle_range[1]-angle_range[0]) + angle_range[0]
        angle2 = np.random.uniform()*(angle_range[1]-angle_range[0]) + angle_range[0]
        rot_mat = getRotationMatrix(angle1, 'y')
        rot_mat2 = getRotationMatrix(-angle2, 'y')
        applied_points = np.dot(points, rot_mat)
        applied_points[:, 2] = 0
        perspectives.append(applied_points)
        projection_mats.append(rot_mat)
        applied_points = np.dot(points, rot_mat2)
        applied_points[:, 2] = 0
        perspectives.append(applied_points)
        projection_mats.append(rot_mat2)
    return perspectives, projection_mats


# End: Alignment and CV Operations

# IDs
def add_col(points):
    '''\
    Adds a column to points
    '''
    return np.append(points, np.zeros((len(points), 1)), 1)

def give_ids(points):
    return [{'data':point, 'id':i} for i,point in enumerate(points)]

def remove_ids(points):
    return np.array([point['data'] for point in points])

# MPSE
def get_dist_weights(hidden_projections, n_points, ndim=None):
    if not ndim:
        ndim = len(hidden_projections[0][0]['data'])
    dist_mats = [np.zeros((n_points, n_points)) for _ in range(len(hidden_projections))]
    weights_mats = [np.zeros((n_points, n_points)) for _ in range(len(hidden_projections))]


    for persp, dist_mat, weights_mat in tqdm(zip(hidden_projections, dist_mats, weights_mats)):
        for point1, point2 in itertools.combinations_with_replacement(persp, 2):
            dist = np.linalg.norm(point1['data'][:ndim] - point2['data'][:ndim])
            dist_mat[point1['id'], point2['id']] = dist
            dist_mat[point2['id'], point1['id']] = dist
            weights_mat[point1['id'], point2['id']] = 1
            weights_mat[point2['id'], point1['id']] = 1
    return dist_mats, weights_mats

def ray_traceZ(points, ray_r=None, n_raysX=None, n_raysY=None):
    '''
    points is a list of dictionaries
    points: [ {'data': ndarray[3], 'id': id} ]
    '''
    def ray_intersect(point, rayxy, r):
        return (point[0] - rayxy[0])*(point[0] - rayxy[0]) + (point[1] - rayxy[1])*(point[1] - rayxy[1]) <= r*r

    only_points = np.array([point['data'] for point in points])
    
    if not n_raysX:
        n_raysX = len(points)//3
    if not n_raysY:
        n_raysY = len(points)//3
        
    xrays = np.linspace(only_points[:, 0].min(), only_points[:, 0].max(), n_raysX)
    yrays = np.linspace(only_points[:, 1].min(), only_points[:, 1].max(), n_raysY)
    
    if not ray_r:
        ray_r = np.min([xrays[1] - xrays[0], yrays[1] - yrays[0]])*2
    
    z_dist = [
        [
            {'data': [np.inf, np.inf, np.inf], 'id': -1} 
            for _ in range(len(xrays))
        ] for _ in range(len(yrays))
    ]

    for i, x in enumerate(xrays):
        for j, y in enumerate(yrays):
            for point in points:
                if ray_intersect(point['data'], (x,y), ray_r):
                    if z_dist[i][j]['data'][2] > point['data'][2]:
                        z_dist[i][j] = point
                    
    new_points = []
    taken_ids = [-1]

    for row in z_dist:
        for point in row:
            if point['id'] not in taken_ids and point['data'][0] != np.inf:
                new_points.append(point)
                taken_ids.append(point['id'])
    return new_points

def show_points_least_persp(weights_mats):
    def points_atleast_n_persp(weights_mats, n):
        ands = []
        for weights in  itertools.combinations(weights_mats, n):
            tmp = np.ones_like(weights_mats[0])
            for i in weights:
                tmp = np.logical_and(tmp, i)
                
            ands.append(tmp)
            
        ret = np.zeros_like(weights_mats[0])
        for i in ands:
            ret = np.logical_or(ret, i)
        n_points = 0
        for row in ret:
            if row.sum() > 0:
                n_points += 1
        return n_points

    points_per_persp = []
    for i in range(1, len(weights_mats)+1):
        points_per_persp.append(
            points_atleast_n_persp(weights_mats, i)
        )
    return points_per_persp

def visible_in_n_persp(labeled_perspectives, points_in_each):
    n_points = len(labeled_perspectives[0])
    n_persp = len(labeled_perspectives)
    to_select = []
    for i in range(n_points):
        to_select.append(np.random.choice(range(n_persp), size=points_in_each, replace=False))
    
    new_labeled_perspectives = [[] for _ in range(n_persp)]
    for i,ts in enumerate(to_select):
        for persp in ts:
            new_labeled_perspectives[persp].append(labeled_perspectives[persp][i])
    for i in range(len(new_labeled_perspectives)):
        new_labeled_perspectives[i] = np.array(new_labeled_perspectives[i])
    return new_labeled_perspectives


def plot_3D(samples_3D, range_x=None, range_y=None, range_z=None, proj_type='perspective', colors=None, eq_range=0, cubic=True, pad=0.4):
    '''\
    Plots sets of 3D points.
    ----------
    samples_3D : List[ndarray], each of shape (n_samples, 3)
        List of Set of points.
    range_x, range_y, range_z : List[int] -> [min, max] 
        The range of the plot.
    proj_type: str
        Type of projection. ('orthographic' or 'perspective')
    Returns
    -------
    fig : go.Figure
        The plotly Figure object
    '''
    plots = []
    if not colors:
        colors = ['green', 'red', 'blue', 'black'][:len(samples_3D)]
        while len(colors) != len(samples_3D):
            colors.append('red')
    assert len(colors) == len(samples_3D), "Color must match the number of samples_3D"
    for sample_3D, color in zip(samples_3D, colors):
        plots.append(go.Scatter3d(
            x=sample_3D[:, 0],
            y=sample_3D[:, 1],
            z=sample_3D[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                opacity=0.5,
                color=color,
                symbol='circle',
            ),
            legendgroup=color,
            showlegend=True,
            name=color
        ))

    axises = []
    
    if not range_x:
        if eq_range is None:
            range_x = [-np.inf, np.inf]
            for i in range(len(samples_3D)):
                range_x = [
                    np.min([range_x[0], samples_3D[i][:, 0].min()]),
                    np.max([range_x[1], samples_3D[i][:, 0].max()])
                ]
        else:
            range_x = [samples_3D[eq_range][:,0].min(), samples_3D[eq_range][:,0].max()]
    if not range_y:
        if eq_range is None:
            range_y = [-np.inf, np.inf]
            for i in range(len(samples_3D)):
                range_y = [
                    np.min([range_y[0], samples_3D[i][:, 1].min()]),
                    np.max([range_y[1], samples_3D[i][:, 1].max()])
                ]
        else:
            range_y = [samples_3D[eq_range][:,1].min(), samples_3D[eq_range][:,1].max()]
    if not range_z:
        if eq_range is None:
            range_z = [np.inf, -np.inf]
            for i in range(len(samples_3D)):
                range_z = [
                    np.min([range_z[0], samples_3D[i][:, 2].max()]),
                    np.max([range_z[1], samples_3D[i][:, 2].min()])
                ]
        else:
            range_z = [samples_3D[eq_range][:,2].max(), samples_3D[eq_range][:,2].min()]
            
    
    if cubic:
        range_x = np.array([np.min([range_x, range_y, range_z]), np.max([range_x, range_y, range_z])])
        range_x += np.abs(range_x)*[-pad, pad]
        range_y = range_x.copy()
        range_z = range_x.copy()[::-1]
        
    axises.append(go.Scatter3d(x=range_x, y=[0, 0], z=[0, 0], mode='lines', showlegend=True, name="Axis", legendgroup="axis"))
    axises.append(go.Scatter3d(x=[0, 0], y=range_y, z=[0, 0], mode='lines', showlegend=False, name="Axis", legendgroup="axis"))
    axises.append(go.Scatter3d(x=[0, 0], y=[0, 0], z=range_z, mode='lines', showlegend=False, name="Axis", legendgroup="axis"))

    layout = go.Layout(
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
        ), autosize=True,
        scene=dict(
            camera=dict(
                up=dict(x=0, y=1, z=0), 
                center=dict(x=0, y=0, z=0),
                eye=dict({'x': 0, 'y': 0, 'z': 1}),
                projection=dict(type=proj_type)
            ),
            xaxis = dict(nticks=4, range=range_x),
            yaxis = dict(nticks=4, range=range_y),
            zaxis = dict(nticks=4, range=range_z),
            aspectmode='cube'
        )
    )
    
    fig = go.Figure(data=[*plots, *axises], layout=layout)
    return fig

# For champher distance
def array2samples_distance(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1 
    """
    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = np.linalg.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance_numpy(array1, array2, scale=100):
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist*scale


def apply_rotation(points, rot_mat):
    return np.dot(points, rot_mat)

def create_rotated_points(points):
    for angle in [45, 90, 90+45, 180, -45, -90, -90-45, -180]:
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        # for rot around x
        rot_mat1 = np.array([
            [1, 0, 0], 
            [0, c, -s],
            [0, s, c]
        ])
        # for rot around y
        rot_mat2 = np.array([
            [c, 0, s], 
            [0, 1, 0], 
            [-s, 0, c]
        ])
        # for rot around z
        rot_mat3 = np.array([
            [c, -s, 0], 
            [s, c, 0], 
            [0, 0, 1]
        ])
        yield from [apply_rotation(points, rot_mat) for rot_mat in [rot_mat1, rot_mat2, rot_mat3]]
        
# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp])
    
# def convert_open3D_pcd(points):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     return pcd

def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


# Calculate reconstruction metrics
def t_to_homo(mat):
    if mat.shape[0] == 3:
        mat = np.append(mat, [[0, 0, 0]], axis=0)
        mat = np.insert(mat, 3, [0, 0, 0, 1], axis=1)
    return mat

def points_to_homo(points):
    if points.shape[1] == 4:
        return points
    return np.append(points, np.ones((len(points), 1)), axis=1)

def points_to_non_homo(points):
    if points.shape[1] == 3:
        return points
    return points[:, :3]

def apply_transformation(points, trans_mat):
    initial_shape = points.shape
    if trans_mat.shape[0] == 3:
        trans_mat = t_to_homo(trans_mat)
    if initial_shape[1] == 3:
        points = points_to_homo(points)
    ret = np.dot(trans_mat, points.T).T
    if initial_shape[1] == 3:
        ret = points_to_non_homo(ret)
    return ret

def get_trans_4_points(points_a, points_b):
    points_a = np.array(points_a)
    points_b = np.array(points_b)
    assert points_a.shape[0] == 4 and points_b.shape[0] == 4, "There must be exactly 4 points"
    return np.dot(np.linalg.inv(points_to_homo(points_a)), points_to_homo(points_b)).T

def get_4pointsample_transform_mat(points, gt_points, dist_agg_fn=None, t_sample=10000):
    if not dist_agg_fn:
        dist_agg_fn = lambda x: np.mean(x)
    
    points = np.array(points)
    gt_points = np.array(gt_points)
    min_dist = np.inf
    best_t = np.identity(4)
    for _ in tqdm(range(t_sample), desc="4 Point Alignment"):
        selected_ids = np.random.choice(range(len(points)), 4)
        sampled_points = points_to_homo(points[selected_ids])
        gt_sampled_points = points_to_homo(gt_points[selected_ids])
        try:
            trans_mat = get_trans_4_points(sampled_points, gt_sampled_points)
        except np.linalg.LinAlgError as e:
            continue
        transformed_points = apply_transformation(points, trans_mat)
        dist = dist_agg_fn(np.linalg.norm(transformed_points - gt_points, axis=1))
        if dist < min_dist:
            min_dist = dist
            best_t = trans_mat
    return best_t, min_dist

def get_optimal_trans_mat(points_a, points_b, iterations=10000, dist_agg_fn=None, lr=0.001):
    if not dist_agg_fn:
        dist_agg_fn = lambda x: tf.reduce_mean(x)
    embeddings_tf = tf.convert_to_tensor(points_to_homo(points_a))
    gt_tf = tf.convert_to_tensor(points_to_homo(points_b))
    transform_mat = tf.Variable(np.identity(4), name='x', trainable=True, dtype=tf.float64)

    @tf.function
    def loss_fn():
        transformed = tf.transpose(tf.tensordot(transform_mat, tf.transpose(embeddings_tf), axes=1))
        dist = tf.norm(transformed - gt_tf, axis=1)
        return dist_agg_fn(dist)


    opt = tf.keras.optimizers.Adam()

    for i in tqdm(range(iterations), desc="Optimal Alignment"):
        opt.minimize(loss_fn, transform_mat)
    return transform_mat.numpy(), loss_fn().numpy()

def get_svd_trans(X, Y, dist_agg_fn=None):
    if not dist_agg_fn:
        dist_agg_fn = lambda x: np.mean(x)
    T = np.linalg.inv(X.T @ X) @ Y.T @ X
    U, Sigma, V = np.linalg.svd(T)
    T = U @ V

    T = t_to_homo(T)
    tmp_embedding = apply_transformation(X, T)
    dist_vec = Y.mean(axis=0) - tmp_embedding.mean(axis=0)
    dist_vec = Y[0] - tmp_embedding[0]
    for i in range(3):
        T[i, 3] = dist_vec[i]
        
    return T, dist_agg_fn(np.linalg.norm(apply_transformation(X,T) - Y, axis=1))

# def get_icp_trans_mat(points_a, points_b, d_th=80, max_iter=1000, max_change_ratio=0.000001):
#     coords_dict = {
#         'Embedding': points_a,
#         'GT': points_b,
#     }
#     # First, we initialize an ICP object. 
#     # The algorithm iteratively matches the ‘k’ closest points. 
#     # To limit the ratio of mismatched points, the ‘radii’ parameter is provided. It defines an ellipsoid within points can be assigned.
#     radii = [d_th, d_th, d_th]
#     icp = registration.ICP(
#         radii,
#         max_iter=max_iter,
#         max_change_ratio=max_change_ratio,
#         # max_change_ratio=0.01,
#         k=1,
#     )

#     T_dict, pairs_dict, report = icp(coords_dict)

#     embedding_trans = np.dot(np.linalg.inv(T_dict['GT']), T_dict['Embedding'])
    
    
#     aligned_a = apply_transformation(points_a, embedding_trans)
    
#     loss = np.linalg.norm(aligned_a - points_b, axis=1)
#     loss = np.sqrt(np.mean(loss*loss))

#     return embedding_trans, loss

def get_icp_trans_mat_by_random_rotation(points_a, points_b, d_th=80, max_iter=1000, max_change_ratio=0.000001):
    best_match = None
    for rotated_points in tqdm(list(reversed(list(create_rotated_points(points_a)))), desc="ICP"):
        coords_dict = {
            'Embedding': rotated_points,
            'GT': points_b,
        }
        # First, we initialize an ICP object. 
        # The algorithm iteratively matches the ‘k’ closest points. 
        # To limit the ratio of mismatched points, the ‘radii’ parameter is provided. It defines an ellipsoid within points can be assigned.
        radii = [d_th, d_th, d_th]
        icp = registration.ICP(
            radii,
            max_iter=max_iter,
            max_change_ratio=max_change_ratio,
            # max_change_ratio=0.01,
            k=1,
        )

        T_dict, pairs_dict, report = icp(coords_dict)

        aligned_embedding1 = transformation.transform(coords_dict['Embedding'], T_dict['Embedding'])
        aligned_GT1 = transformation.transform(coords_dict['GT'], T_dict['GT'])

        if not best_match or report['RMSE'][-1] < best_match['RMSE']:
            best_match = {
                'RMSE': report['RMSE'][-1],
                'aligned_points_a': aligned_embedding1,
                'aligned_points_b': aligned_GT1,
                'trans_mat_points_a': T_dict['Embedding'],
                'trans_mat_points_b': T_dict['GT'],
            }
            
    return best_match