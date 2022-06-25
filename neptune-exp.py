#!/usr/bin/env python
# coding: utf-8

import neptune.new as neptune
from neptune.new.types import File

from matplotlib import pyplot as plt
from plotly.io import to_html
import numpy as np
seed = np.random.randint(1, 500)
np.random.seed(seed)
from tqdm import tqdm
import time

import copy

from MPSE.MPSE import mview
from utils import *

import pprint as pp

def upload_numpy(run, data, location):
    upload_numpy.counter += 1
    file_name = f"/tmp/{upload_numpy.counter}.npy"
    np.save(file_name, data)
    run[location].upload(file_name)
upload_numpy.counter = 0


# Parameters
EXPERIMENT_NAME_PREFIX = "MPSE-mds"
init_params = dict(
    # DATASET = 'toy_points',
    DATASET = 'ModelNet10:desk:0013',
    INITIAL_EMBEDDING = False,
    N_POINTS = 256,
    N_PERSPECTIVE = 10,
    N_PROJECTION_DIM = 2,
    PROJECTION = dict(
        PROJ_TYPE = 'atleast_in_n_persp',
        POINT_IN_ATLEAST = 8,
    ),
    # PROJECTION = dict(
    #     PROJ_TYPE = 'raytracing',
    #     N_XRAYS = 200,
    #     N_YRAYS = 200,
    # ),
    MPSE = dict(
        BATCH_SIZE = 256,
        MAX_ITER = 300,
        MIN_GRAD = 1e-10,
        MIN_COST = 1e-10,
        VERBOSE = 2,
        SMART_INITIALIZATION = True,
        INITIAL_PROJECTIONS = 'cylinder',
        VARIABLE_PROJECTION = False
    ),
    NUMPY_SEED = seed,
    UPLOAD_LARGE_VIZ = False,
    tags = []
)
all_params = [init_params]

# Add all the run configs
# datasets_to_take = [
#     # 'ModelNet10:desk:0005', 
#     'ModelNet10:chair:0001',
#     'ModelNet10:toilet:0001',
#     'ModelNet10:table:0002',
#     'ModelNet10:sofa:0019', 
#     'ModelNet10:night_stand:0010',
#     'ModelNet10:monitor:0016', 
#     # 'ModelNet10:monitor:0003',
#     # 'ModelNet10:dresser:0001', 
#     # 'ModelNet10:bathtub:0050',
#     # 'ModelNet10:bathtub:0005', 
#     # 'ModelNet10:bed:0003',
#     # 'ModelNet10:bed:0005', 
#     'ModelNet10:bed:0001',
#     'ModelNet10:desk:0013', 
#     # 'ModelNet10:desk:0006'
# ]


template_params = copy.deepcopy(all_params.pop(0))
for _ in range(5):
    for n_points in [256, 512]:
        new_param = copy.deepcopy(template_params)
        new_param['N_POINTS'] = n_points
        all_params.append(new_param)

# for _ in range(30):
#     for n_persp in [4, 8, 12, 14]:
#     # for n_persp in [12, 14]:
#         new_param_tmp = copy.deepcopy(init_params)
#         new_param_tmp['N_PERSPECTIVE'] = n_persp
#         for i in range(1, n_persp+1):
#             new_param_tmp2 = copy.deepcopy(new_param_tmp)
#             new_param_tmp2['PROJECTION']['POINT_IN_ATLEAST'] = i
#             for proj in [True, False]:
#                 new_param = copy.deepcopy(new_param_tmp2)
#                 new_param['MPSE']['VARIABLE_PROJECTION'] = proj
#                 all_params.append(new_param)


# template_params = copy.deepcopy(all_params.pop(0))
# for _ in range(1):
#     for dataset in datasets_to_take:
#         new_params_dataset = copy.deepcopy(template_params)
#         new_params_dataset['DATASET'] = dataset
#         for n_persp in range(2, 13):
#             new_param_tmp = copy.deepcopy(new_params_dataset)
#             new_param_tmp['N_PERSPECTIVE'] = n_persp
#             for p_in_persp in range(1, n_persp+1):
#                 new2 = copy.deepcopy(new_param_tmp)
#                 new2['PROJECTION']['POINT_IN_ATLEAST'] = p_in_persp
#                 for proj in [True, False]:
#                     new_param2 = copy.deepcopy(new2)
#                     new_param2['MPSE']['VARIABLE_PROJECTION'] = proj
#                     all_params.append(new_param2)

        # n_persp = new_params_dataset['N_PERSPECTIVE']
        # for i in range(1, n_persp+1):
        #     new_param_tmp = copy.deepcopy(new_params_dataset)
        #     new_param_tmp['PROJECTION']['POINT_IN_ATLEAST'] = i
        # for n_points in [100, 200, 300, 400, 500]:
        #     new_param = copy.deepcopy(new_params_dataset)
        #     new_param['N_POINTS'] = n_points
        #     for proj in [True, False]:
        #         new_param2 = copy.deepcopy(new_param)
        #         new_param2['MPSE']['VARIABLE_PROJECTION'] = proj
        #         all_params.append(new_param2)

for params in all_params:
    upload_numpy.counter = 0
    params['tags'] = params['tags'] + [str(params['DATASET']), str(params['N_PERSPECTIVE']), str(params['PROJECTION']['PROJ_TYPE']), 'it2']
    # Assertions
    if params['PROJECTION']['PROJ_TYPE'] == "atleast_in_n_persp":
        assert 1 <= params['PROJECTION']['POINT_IN_ATLEAST'] <= params['N_PERSPECTIVE'], "0 <= POINT_IN_ATLEAST <= N_PERSPECTIVE"

    EXPERIMENT_NAME = f"{EXPERIMENT_NAME_PREFIX}-" if EXPERIMENT_NAME_PREFIX else ""
    EXPERIMENT_NAME += params['DATASET']

    pp.pprint(EXPERIMENT_NAME)
    pp.pprint(params)

    print("Loading Dataset...")
    points = get_dataset_points(params['DATASET'], params['N_POINTS'])
    params['N_POINTS'] = len(points)

    ## Neptune
    run = neptune.init(
        project="rahatzamancse/MPSE-mds",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3NDk5MWVkNS0wMDg4LTRmNTktYWQyZC0zNzUyNTg0MTY1MGYifQ==",
        source_files=["neptune-exp.py", "utils.py"],
        capture_hardware_metrics=False,
        capture_stdout=True,
        capture_stderr=True,
        tags=params['tags']
    )
    run["parameters"] = params

    upload_numpy(run, points, 'GT/X')
    if params['UPLOAD_LARGE_VIZ']:
        # Save the plot
        fig = plot_3D([points])
        html_str = to_html(fig)
        run['GT'].upload(File.from_content(
            html_str, extension='html'
        ))

    labeled_perspectives = []
    perspectives, projection_mats = get_randomized_all_persps(points, params['N_PERSPECTIVE'])
    projection_mats = np.array(projection_mats)
    for perspective in perspectives:
        labeled_perspectives.append(give_ids(perspective))

    if params['PROJECTION']['PROJ_TYPE'] == 'atleast_in_n_persp':
        labeled_perspectives = visible_in_n_persp(labeled_perspectives, params['PROJECTION']['POINT_IN_ATLEAST'])
    elif params['PROJECTION']['PROJ_TYPE'] == 'raytracing':
        labeled_perspectives = [
            ray_traceZ(p, n_raysX=params['PROJECTION']['N_XRAYS'], n_raysY=params['PROJECTION']['N_YRAYS']) for p in tqdm(labeled_perspectives)
        ]

    for i, view in enumerate(labeled_perspectives):
        image_dim = np.array([500, 500])
        view = remove_ids(view)[:, :2]
        view = (view - view.min(axis=0))/(view.max(axis=0) - view.min(axis=0)) * np.array(image_dim - 1)
        view = view.astype(np.uint)
        image = np.zeros(image_dim)
        image[view[:, 0], view[:,1]] = 1
        run[f'GT/perspectives/{i}'].upload(File.as_image(image))

    dist_mats, weights_mats = get_dist_weights(labeled_perspectives, len(points), params['N_PROJECTION_DIM'])

    points_per_persp = np.array(show_points_least_persp(weights_mats))
    points_per_persp = points_per_persp/points_per_persp.max() * 100

    run['GT/Points per perspective list'] = ' '.join(str(i) for i in points_per_persp)

    projection_kwargs = {}
    if params['MPSE']['VARIABLE_PROJECTION']:
        projection_kwargs['initial_projections'] = params['MPSE']['INITIAL_PROJECTIONS']
        projection_kwargs['fixed_projections'] = None
    else:
        projection_kwargs['fixed_projections'] = projection_mats

    start = time.time()
    mv = mview.basic(
        dist_mats.copy(),
        batch_size = params['MPSE']['BATCH_SIZE'],
        max_iter=params['MPSE']['MAX_ITER'],
        min_grad=params['MPSE']['MIN_GRAD'],
        min_cost=params['MPSE']['MIN_COST'],
        verbose=params['MPSE']['VERBOSE'], 
        smart_initialization=params['MPSE']['SMART_INITIALIZATION'],
        weights=weights_mats.copy(),
        initial_embedding = np.random.uniform(points.min(), points.max(), (params['N_POINTS'], 3)) if params['INITIAL_EMBEDDING'] else None,
        **projection_kwargs
    )
    end = time.time()

    run['runtime'] = end - start

    run['Results/computation history/Conclusion'] = mv.computation_history[0]['conclusion']
    run['Results/computation history/Actual Iterations Run'] = mv.computation_history[0]['iterations']

    run['Results/Final Cost'] = mv.cost
    run['Results/Final Individual Cost'] = mv.individual_cost

    for item in ['grads', 'costs', 'steps', 'lrs']:
        for val in mv.computation_history[0][item]:
            run[f'Results/computation history/{item}'].log(val)

    # Plots from MPSE mview
    fig = mv.plot_computations()
    run['Results/Computations'].upload(fig)
    plt.close(fig)
    # fig = mv.plot_embedding()
    # run['Results/MPSE_embedding'].upload(fig)
    # plt.close(fig)
    fig = mv.plot_images()
    run['Results/Perspectives'].upload(fig)
    plt.close(fig)
    upload_numpy(run, mv.projections, 'Results/Final Projections')
    upload_numpy(run, np.array(projection_mats), 'GT/Projections')

    embeddings = mv.X.copy()
    if params['UPLOAD_LARGE_VIZ']:
        run['Results/Embedding_vis'].upload(File.as_html(
            plot_3D([embeddings])
        ))
    upload_numpy(run, embeddings, 'Results/Embedding_X')

    # Point Alignment
    print("4 point sample Alignment method:")
    trans_mat, best_dist = get_4pointsample_transform_mat(embeddings, points)
    print("Best Distance :", best_dist)
    print("Transformation matrix :\n", trans_mat)
    prefix = 'Results/Alignment/4PointSample/'
    aligned_embeddings = apply_transformation(embeddings, trans_mat)
    if params['UPLOAD_LARGE_VIZ']:
        run[prefix+'Vis'].upload(File.as_html(
            plot_3D([points, aligned_embeddings], colors=['green', 'red'])
        ))
    upload_numpy(run, trans_mat, prefix+'transformation_mat')
    run[prefix+'error'] = best_dist



    range_x = [
        points[:, 0].min(),
        points[:, 0].max()
    ]
    range_y = [
        points[:, 1].min(),
        points[:, 1].max()
    ]
    range_z = [
        points[:, 2].min(),
        points[:, 2].max()
    ]
    
    d_th = max([r[1] - r[0] for r in [range_x, range_y, range_z]])

    # 4 point alignment + ICP
    # icp_trans_mat, icp_loss = get_icp_trans_mat(aligned_embeddings, points, d_th=d_th, max_iter=1000)
    # prefix = 'Results/Alignment/4Point_ICP/'
    # if params['UPLOAD_LARGE_VIZ']:
    #     run[prefix+'Vis'].upload(File.as_html(
    #         plot_3D([
    #                 points,
    #                 apply_transformation(
    #                     apply_transformation(embeddings, trans_mat), 
    #                     icp_trans_mat
    #                 )
    #             ], 
    #             colors=['green', 'red']
    #         )
    #     ))
    # upload_numpy(run, icp_trans_mat, prefix + 'transformation_mat')
    # run[prefix+'error'] = icp_loss
    
    print("Global RMSE optimization method:")
    trans_mat, loss = get_optimal_trans_mat(embeddings, points)
    prefix = 'Results/Alignment/Global-RMSE-opt/'
    aligned_embeddings = apply_transformation(embeddings, trans_mat)
    if params['UPLOAD_LARGE_VIZ']:
        run[prefix+'Vis'].upload(File.as_html(
            plot_3D([points, aligned_embeddings], colors=['green', 'red'])
        ))
    upload_numpy(run, trans_mat, prefix+'transformation_mat')
    run[prefix+'error'] = loss

    # Global RMSE + ICP
    # icp_loss, icp_trans_mat = get_icp_trans_mat(aligned_embeddings, points, d_th=d_th, max_iter=1000)
    # prefix = 'Results/Alignment/GlobalRMSE_ICP/'
    # if params['UPLOAD_LARGE_VIZ']:
    #     run[prefix+'Vis'].upload(File.as_html(
    #         plot_3D([
    #                 points,
    #                 apply_transformation(
    #                     apply_transformation(embeddings, trans_mat), 
    #                     icp_trans_mat
    #                 )
    #             ], 
    #             colors=['green', 'red']
    #         )
    #     ))
    # upload_numpy(run, icp_trans_mat, prefix + 'transformation_mat')
    # run[prefix+'error'] = icp_loss

    plt.close('all')
    run.stop()