#!/usr/bin/env python
# coding: utf-8

import neptune.new as neptune
from neptune.new.types import File

from matplotlib import pyplot as plt
from plotly.io import to_html
import numpy as np
from tqdm import tqdm
import time

import copy

from MPSE.MPSE import mview
from utils import *

from rich.console import Console
from rich.progress import track
console = Console()

pix3d_datadir = '/run/media/insane/My 4TB 2/Big Data/MPSE/Data/pix3d/'
shapenet_datadir = '/run/media/insane/My 4TB 2/Big Data/MPSE/Data/ShapeNetCore/ShapeNetCore.v2'
modelnet_datadir = None

def upload_numpy(run, data, location):
    upload_numpy.counter += 1
    file_name = f"/tmp/{upload_numpy.counter}.npy"
    np.save(file_name, data)
    run[location].upload(file_name)
upload_numpy.counter = 0


# Parameters
EXPERIMENT_NAME_PREFIX = "AngleRange"
init_params = dict(
    # DATASET = 'toy_points',
    # DATASET = 'ModelNet10:desk:0013',
    DATASET = 'Pix3D:chair:0001',
    # DATA_DIR = pix3d_datadir,
    DATA_DIR = shapenet_datadir,
    NORMALIZE_DATASET_POINTS = True,
    INITIAL_EMBEDDING = False,
    # NOISE_TYPE='permutation',
    NOISE_TYPE='distance',
    NOISE_AMOUNT = 0.0,
    NOISE_AMP = 0.0,
    N_POINTS = 512,
    N_PERSPECTIVE = 5,
    N_PROJECTION_DIM = 2,
    ANGLE_RANGE = dict(
        START=0,
        END=360
    ),
    PROJECTION = dict(
        PROJ_TYPE = 'atleast_in_n_persp',
        POINT_IN_ATLEAST = 4,
    ),
    # PROJECTION = dict(
    #     PROJ_TYPE = 'raytracing',
    #     N_XRAYS = 600,
    #     N_YRAYS = 600,
    # ),
    MPSE = dict(
        BATCH_SIZE = 2048,
        MAX_ITER = 300,
        MIN_GRAD = 1e-4,
        MIN_COST = 1e-4,
        VERBOSE = 0,
        SMART_INITIALIZATION = True,
        INITIAL_PROJECTIONS = 'cylinder',
        VARIABLE_PROJECTION = True
    ),
    # NUMPY_SEED = seed,
    UPLOAD_LARGE_VIZ = False,
    UPLOAD_MPSE_FIGS = False,
    tags = ['angle-variable-projection']
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

pix3D_dataset = [
    'Pix3D:chair:0132',
    'Pix3D:chair:1582',
]

datasets_to_take = [
    'ShapeNet:airplane:103c9e43cdf6501c62b600da24e0965',
    'ShapeNet:airplane:105f7f51e4140ee4b6b87e72ead132ed',
    'ShapeNet:airplane:10e4331c34d610dacc14f1e6f4f4f49b',
    'ShapeNet:airplane:d405b9e5f942fed5efe5d5ae25ee424e',
    'ShapeNet:airplane:157bb84c08754307dff9b4d1071b12d7',
    'ShapeNet:airplane:8cf06a71987992cf90a51833252023c7',

    'ShapeNet:bench:42ffe8d78c2e8da9d40c07d3c15cc681',
    'ShapeNet:bench:cad0a0e60708ab662ab293e158725cf0',
    'ShapeNet:bench:cca18c7f8636606f51f77a6d7299806',
    'ShapeNet:bench:89e2eaeb437cd42f85e40cb3507a0145',
    'ShapeNet:bench:702870d836ea3bf32056b4bd5d870b47',
    'ShapeNet:bench:fc0486ec53630bdbd2b12aa6a0f050b3',

    'ShapeNet:car:44f30f4c65c3142a16abce8cb03e7794',
    'ShapeNet:car:d9034b15c7f295551a46c391b387480b',
    'ShapeNet:car:35de0d0cc71179dc1a98dff5b6c5dec6',
    'ShapeNet:car:d6f8cfdb1659142814fccfc8a25361e',
    'ShapeNet:car:d79f66a4566ff981424db5a60837de26',

    'ShapeNet:chair:bf91d0169eae3bfdd810b14a81e12eca',
    'ShapeNet:chair:6a3d2feff3783804387379bbd607d69e',
    'ShapeNet:chair:cd6a8020b69455dbb161f36d4e309050',
    'ShapeNet:chair:cd9702520ad57689bbc7a6acbd8f058b',

    'ShapeNet:lamp:102273fdf8d1b90041fbc1e2da054acb',
    'ShapeNet:lamp:fa0a32c4326a42fef51f77a6d7299806',
    'ShapeNet:lamp:e6d62a37e187bde599284d844aba7576',

    'ShapeNet:rifle:10cc9af8877d795c93c9577cd4b35faa',
    'ShapeNet:rifle:81ba8d540499dd04834bde3f2f2e7c0c',
    'ShapeNet:rifle:823b97177d57e5dd8e0bef156e045efe',
    'ShapeNet:rifle:f55544d331eb019a1aca20a2bd5ca645',

    'ShapeNet:table:105b9a03ddfaf5c5e7828dbf1991f6a4',
    'ShapeNet:table:c3884d2d31ac0ac9593ebeeedbff73b',
    'ShapeNet:table:16961ddf69b6e91ea8ff4f6e9563bff6',
    'ShapeNet:table:86e6ef5ae3420e95963080fd7249126d',

    'ShapeNet:sofa:79bea3f7c72e0aae490ad276cd2af3a4',
    'ShapeNet:sofa:cff485b2c98410135dda488a4bbb1e1',
    'ShapeNet:sofa:d5a2b159a5fbbc4c510e2ce46c1af6e',
    'ShapeNet:sofa:d8c748ced5e5f2cc7e3820d17093b7c2'
]

dataset_tags = {
    'ShapeNet:airplane:103c9e43cdf6501c62b600da24e0965': 'S1-1',
    'ShapeNet:airplane:105f7f51e4140ee4b6b87e72ead132ed': 'S1-6',
    'ShapeNet:airplane:10e4331c34d610dacc14f1e6f4f4f49b': '3-2',
    'ShapeNet:airplane:d405b9e5f942fed5efe5d5ae25ee424e': 'S1-1',
    'ShapeNet:airplane:157bb84c08754307dff9b4d1071b12d7': 'S1-3',
    'ShapeNet:airplane:8cf06a71987992cf90a51833252023c7': '3-1',

    'ShapeNet:bench:42ffe8d78c2e8da9d40c07d3c15cc681': '3-4',
    'ShapeNet:bench:cad0a0e60708ab662ab293e158725cf0': 'S1-8',
    'ShapeNet:bench:cca18c7f8636606f51f77a6d7299806': 'S1-7',
    'ShapeNet:bench:89e2eaeb437cd42f85e40cb3507a0145': 'S1-12',
    'ShapeNet:bench:702870d836ea3bf32056b4bd5d870b47': 'S1-11',
    'ShapeNet:bench:fc0486ec53630bdbd2b12aa6a0f050b3': 'S1-10',

    'ShapeNet:car:44f30f4c65c3142a16abce8cb03e7794': '3-6',
    'ShapeNet:car:d9034b15c7f295551a46c391b387480b': 'S1-17',
    'ShapeNet:car:35de0d0cc71179dc1a98dff5b6c5dec6': 'S1-13',
    'ShapeNet:car:d6f8cfdb1659142814fccfc8a25361e': 'S1-16',
    'ShapeNet:car:d79f66a4566ff981424db5a60837de26': 'S1-15',

    'ShapeNet:chair:bf91d0169eae3bfdd810b14a81e12eca': '3-8',
    'ShapeNet:chair:6a3d2feff3783804387379bbd607d69e': '1-7',
    'ShapeNet:chair:cd6a8020b69455dbb161f36d4e309050': 'S1-20',
    'ShapeNet:chair:cd9702520ad57689bbc7a6acbd8f058b': 'S1-21',

    'ShapeNet:lamp:102273fdf8d1b90041fbc1e2da054acb': '3-9',
    'ShapeNet:lamp:fa0a32c4326a42fef51f77a6d7299806': 'S2-19',
    'ShapeNet:lamp:e6d62a37e187bde599284d844aba7576': 'S2-18',

    'ShapeNet:rifle:10cc9af8877d795c93c9577cd4b35faa': 'S2-2',
    'ShapeNet:rifle:81ba8d540499dd04834bde3f2f2e7c0c': 'S2-4',
    'ShapeNet:rifle:823b97177d57e5dd8e0bef156e045efe': '3-10',
    'ShapeNet:rifle:f55544d331eb019a1aca20a2bd5ca645': 'S2-3',

    'ShapeNet:table:105b9a03ddfaf5c5e7828dbf1991f6a4': 'S2-11',
    'ShapeNet:table:c3884d2d31ac0ac9593ebeeedbff73b': 'S2-15',
    'ShapeNet:table:16961ddf69b6e91ea8ff4f6e9563bff6': 'S2-16',
    'ShapeNet:table:86e6ef5ae3420e95963080fd7249126d': 'S2-14',

    'ShapeNet:sofa:79bea3f7c72e0aae490ad276cd2af3a4': 'S2-8',
    'ShapeNet:sofa:cff485b2c98410135dda488a4bbb1e1': 'S2-5',
    'ShapeNet:sofa:d5a2b159a5fbbc4c510e2ce46c1af6e': 'S2-6',
    'ShapeNet:sofa:d8c748ced5e5f2cc7e3820d17093b7c2': '3-5',
}

project = neptune.get_project(
    name='rahatzamancse/3DMPE-angle-noise', 
    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3NDk5MWVkNS0wMDg4LTRmNTktYWQyZC0zNzUyNTg0MTY1MGYifQ=='
)

run_table_df = project.fetch_runs_table(
    owner='rahatzamancse',
).to_pandas()

new_param = copy.deepcopy(all_params.pop(0))
MIN_NUMBER_OF_RUN_REQUIRED = 4
for dataset in [
        'ModelNet10:chair:0001',
        'ShapeNet:airplane:103c9e43cdf6501c62b600da24e0965',
        'ShapeNet:table:105b9a03ddfaf5c5e7828dbf1991f6a4',
        'ShapeNet:sofa:79bea3f7c72e0aae490ad276cd2af3a4',
        'ShapeNet:rifle:81ba8d540499dd04834bde3f2f2e7c0c',
    ]:
    new_param = copy.deepcopy(new_param)
    new_param['DATASET'] = dataset
    
    # for smart_init in [True, False]:
    #     new_param = copy.deepcopy(new_param)
    #     new_param['MPSE']['SMART_INITIALIZATION'] = smart_init

    for angle_range in map(lambda end: {'START': 0, 'END': end}, [360, 180, 90, 60, 45]):
        new_param = copy.deepcopy(new_param)
        new_param['ANGLE_RANGE'] = angle_range
    # for noise_amount in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     new_param = copy.deepcopy(new_param)
    #     new_param['NOISE_AMOUNT'] = noise_amount
    
    #     # for noise_amp in [0.01, 0.03, 0.05, 0.09]:
    #     for noise_amp in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    #         new_param = copy.deepcopy(new_param)
    #         new_param['NOISE_AMP'] = noise_amp

        for viewpoints in [5]:
            new_param = copy.deepcopy(new_param)
            new_param['N_PERSPECTIVE'] = viewpoints
            for points_visible in [4]:
                new_param = copy.deepcopy(new_param)
                new_param['PROJECTION']['POINT_IN_ATLEAST'] = points_visible
                
                existing_runs = run_table_df[
                    (run_table_df['parameters/DATASET'] == new_param['DATASET'])
                    & (run_table_df['parameters/INITIAL_EMBEDDING'] == new_param['INITIAL_EMBEDDING'])
                    & (run_table_df['parameters/MPSE/BATCH_SIZE'] == new_param['MPSE']['BATCH_SIZE'])
                    & (run_table_df['parameters/MPSE/SMART_INITIALIZATION'] == new_param['MPSE']['SMART_INITIALIZATION'])
                    & (run_table_df['parameters/MPSE/VARIABLE_PROJECTION'] == new_param['MPSE']['VARIABLE_PROJECTION'])
                    & (run_table_df['parameters/N_POINTS'] == new_param['N_POINTS'])
                    & (run_table_df['parameters/N_PERSPECTIVE'] == new_param['N_PERSPECTIVE'])
                    & (run_table_df['parameters/PROJECTION/POINT_IN_ATLEAST'] == new_param['PROJECTION']['POINT_IN_ATLEAST'])
                    & (run_table_df['parameters/PROJECTION/PROJ_TYPE'] == new_param['PROJECTION']['PROJ_TYPE'])
                    & (run_table_df['parameters/NOISE_AMOUNT'] == new_param['NOISE_AMOUNT'])
                    & (run_table_df['parameters/ANGLE_RANGE/START'] == new_param['ANGLE_RANGE']['START'])
                    & (run_table_df['parameters/ANGLE_RANGE/END'] == new_param['ANGLE_RANGE']['END'])
                    & (run_table_df['parameters/NOISE_AMP'] == new_param['NOISE_AMP'])
                    & (run_table_df['parameters/NOISE_TYPE']== new_param['NOISE_TYPE'])
                ]
                
                for ei in range(MIN_NUMBER_OF_RUN_REQUIRED - len(existing_runs)):
                    seed = np.random.randint(1, 500)
                    np.random.seed(seed)
                    new_param['NUMPY_SEED'] = seed
                    all_params.append(new_param)
                    # if this is the last iteration, break
                    if ei == MIN_NUMBER_OF_RUN_REQUIRED - len(existing_runs) - 1:
                        break
                else:
                    print("Skipping this run as it already exists in experiment:", existing_runs['sys/id'].tolist())

console.print(f"Total experiments to run: {len(all_params)}\n", style="bold red")

for exp_i, params in track(enumerate(all_params), total=len(all_params), transient=True):
    console.print(f"Running experiment:{exp_i}/{len(all_params)}", style="bold red")
    upload_numpy.counter = 0
    params['tags'] = params['tags'] + []
    # Assertions
    if params['PROJECTION']['PROJ_TYPE'] == "atleast_in_n_persp":
        assert 1 <= params['PROJECTION']['POINT_IN_ATLEAST'] <= params['N_PERSPECTIVE'], "0 <= POINT_IN_ATLEAST <= N_PERSPECTIVE"

    EXPERIMENT_NAME = f"{EXPERIMENT_NAME_PREFIX}-" if EXPERIMENT_NAME_PREFIX else ""
    EXPERIMENT_NAME += params['DATASET']

    console.log(EXPERIMENT_NAME)
    console.log(params)

    print("Loading Dataset...")
    points = get_dataset_points(params['DATASET'], n_points=params['N_POINTS'], datadir=params['DATA_DIR'], normalize=params['NORMALIZE_DATASET_POINTS'])
    params['N_POINTS'] = len(points)

    ## Neptune
    run = neptune.init(
        project="rahatzamancse/3DMPE-angle-noise",
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
    perspectives, projection_mats = get_randomized_all_persps(points, params['N_PERSPECTIVE'], [params['ANGLE_RANGE']['START'], params['ANGLE_RANGE']['END']])
    projection_mats = np.array(projection_mats)
    for perspective in perspectives:
        labeled_perspectives.append(give_ids(perspective))

    if params['PROJECTION']['PROJ_TYPE'] == 'atleast_in_n_persp':
        labeled_perspectives = visible_in_n_persp(labeled_perspectives, params['PROJECTION']['POINT_IN_ATLEAST'])
    elif params['PROJECTION']['PROJ_TYPE'] == 'raytracing':
        labeled_perspectives = [
            ray_traceZ(p, n_raysX=params['PROJECTION']['N_XRAYS'], n_raysY=params['PROJECTION']['N_YRAYS']) for p in tqdm(labeled_perspectives)
        ]

    for labeled_perspective in labeled_perspectives:
        for p in labeled_perspective:
            p['data'][2] = 0

    for i, view in enumerate(labeled_perspectives):
        image_dim = np.array([500, 500])
        view = remove_ids(view)[:, :2]
        view = (view - view.min(axis=0))/(view.max(axis=0) - view.min(axis=0)) * np.array(image_dim - 1)
        view = view.astype(np.uint)
        image = np.zeros(image_dim)
        image[view[:, 0], view[:,1]] = 1
        run[f'GT/perspectives/{i}'].upload(File.as_image(image))

    dist_mats, weights_mats = get_dist_weights(labeled_perspectives, len(points), params['N_PROJECTION_DIM'])
    
    dist_mats = add_noise(dist_mats, params['NOISE_AMOUNT'], params['NOISE_AMP'])

    baseline = get_baseline_metrics(points, dist_mats)
    run['Results/Baseline/4point_ICP_Chamfer'] = baseline['chamfer']
    run['Results/Baseline/4point_ICP_EMD'] = baseline['EMD']

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
    if params['UPLOAD_MPSE_FIGS']:
        fig = mv.plot_computations()
        run['Results/Computations'].upload(fig)
        plt.close(fig)

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

    ## Metrics

    # Point Alignment
    # print("4 point sample Alignment method:")
    # trans_mat, best_dist = get_4pointsample_transform_mat(embeddings, points)
    # print("Best Distance :", best_dist)
    # print("Transformation matrix :\n", trans_mat)
    # prefix = 'Results/Alignment/4PointSample/'
    # aligned_embeddings = apply_transformation(embeddings, trans_mat)
    # if params['UPLOAD_LARGE_VIZ']:
    #     run[prefix+'Vis'].upload(File.as_html(
    #         plot_3D([points, aligned_embeddings], colors=['green', 'red'])
    #     ))
    # upload_numpy(run, trans_mat, prefix+'transformation_mat')
    # run[prefix+'error'] = best_dist

    # 4 point alignment + ICP
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

    trans_mat, loss = get_4pointsample_transform_mat(embeddings, points)
    prefix = 'Results/Alignment/ICP/'
    if params['UPLOAD_LARGE_VIZ']:
        icp_trans_mat, icp_loss = get_icp_trans_mat_by_random_rotation(embeddings, points, d_th=d_th, max_iter=1000)
        run[prefix+'Vis'].upload(File.as_html(
            plot_3D([
                    points,
                    apply_transformation(
                        apply_transformation(embeddings, trans_mat), 
                        icp_trans_mat
                    )
                ], 
                colors=['green', 'red']
            )
        ))
    upload_numpy(run, trans_mat, prefix + 'transformation_mat')
    run[prefix+'error'] = loss
    champ_dist = chamfer_distance_numpy(np.expand_dims(apply_transformation(embeddings, trans_mat), axis=0), np.expand_dims(points, axis=0))
    run[prefix+'chamfer_distancex100'] = champ_dist
    run[prefix+'EMDx100'] = emd(apply_transformation(embeddings, trans_mat), points)

    
    # print("Global RMSE optimization method:")
    # trans_mat, loss = get_optimal_trans_mat(embeddings, points)
    # prefix = 'Results/Alignment/Global-RMSE-opt/'
    # aligned_embeddings = apply_transformation(embeddings, trans_mat)
    # if params['UPLOAD_LARGE_VIZ']:
    #     run[prefix+'Vis'].upload(File.as_html(
    #         plot_3D([points, aligned_embeddings], colors=['green', 'red'])
    #     ))
    # upload_numpy(run, trans_mat, prefix+'transformation_mat')
    # run[prefix+'error'] = loss

    run.sync(wait=True)
    plt.close('all')
    run.stop()