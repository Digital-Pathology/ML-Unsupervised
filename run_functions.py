
import json
import os
import multiprocessing

import torch

from dataset import Dataset, LabelManager, label_extractor
from filtration import FilterManager, FilterBlackAndWhite, FilterHSV, FilterFocusMeasure
from histolab.scorer import NucleiScorer
from histolab.tile import Tile
import PIL
from tqdm import tqdm as loadingbar

from model_analysis.by_region_analysis_job import WeightRatioAnalysisJob
import sagemaker_stuff
from tiling.tile_dataset import TilesDataset
from util import open_file


def announce(*args, **kwargs):
    print(f" \n{kwargs.get('prefix','')}{sagemaker_stuff.config.ANNOUNCEMENT_PREFIX}",
          *args, " \n", **kwargs)


def announce_testing_status():
    if sagemaker_stuff.config.IS_TESTING_LOCALLY:
        announce(f"Test Run!")


def pull_dataset_filtraton_cache():
    sagemaker_stuff.util.download_file_from_s3(
        s3_bucket="digpath-cache",
        s3_path="kevin_supervised/filtration_cache.h5",
        destination='.',
        print_status=True
    )


def pull_tiles_dataset_scoring_data(data_path):
    sagemaker_stuff.util.download_file_from_s3(
        s3_bucket="digpath-tilescore",
        s3_path=data_path,
        destination='.',
        print_status=True
    )


def initialize_dataset():
    # filtration cache
    if sagemaker_stuff.config.FILTRATION_CACHE_DOWNLOAD:
        pull_dataset_filtraton_cache()

    # dataset details
    data_path = os.path.join(
        sagemaker_stuff.config.DIR_DATA_TRAIN,
        sagemaker_stuff.config.DIR_DATA_TRAIN_SUBDIR
    )
    label_map = {
        "mild": 0,
        "moderate": 1,
        "severe": 2
    }
    labels = LabelManager(
        path=data_path,
        label_postprocessor=lambda relpath: label_map[os.path.basename(
            relpath).lower()]
    )
    filtration = FilterManager([
        FilterBlackAndWhite(),
        FilterHSV(),
        FilterFocusMeasure()
    ])
    dataset = Dataset(
        data=data_path,
        labels=labels,
        filtration=filtration
    )

    # tiles dataset
    if sagemaker_stuff.config.DATASET_AS_TILES_DATASET:
        if sagemaker_stuff.config.TILE_SCORING_DATA_DOWNLOAD:
            pull_tiles_dataset_scoring_data(
                sagemaker_stuff.config.TILE_SCORING_DATA_PATH)
        scoring_data_path = "scoring_data.json"
        dataset = TilesDataset(dataset, scoring_data_path)
        sagemaker_stuff.util.copy_file_to_tar_dir(scoring_data_path)

    # save the filtration cache to output dir
    if sagemaker_stuff.config.FILTRATION_CACHE_UPLOAD:
        sagemaker_stuff.util.copy_file_to_tar_dir(
            dataset.filtration_cache.h5filepath)
    # save dataset labels
    if sagemaker_stuff.config.DATASET_SAVE_LABELS:
        save_labels_from_dataset(dataset)
    return dataset


def get_model_callback():
    #model_name = os.environ["MODEL_NAME"]
    ##mm = ModelManager(os.path.join(os.path.dirname(__file__), "models"))
    #mm = ModelManagerForWebApp()
    #m = mm.load_model(model_name)
    # return m.daignose_region
    import torch
    import torchvision
    from lazy_model import my_model
    model_densenet = torchvision.models.DenseNet(
        growth_rate=32,
        block_config=(2, 2, 2, 2),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        num_classes=3
    ).to(torch.device(sagemaker_stuff.config.DATA_AND_MODEL_DEVICE))
    model_densenet.eval()
    model_mymodel = my_model.MyModel(
        model=model_densenet,
        loss_fn=None,
        device=torch.device(sagemaker_stuff.config.DATA_AND_MODEL_DEVICE),
        checkpoint_dir=None,
        optimizer=None,
        model_dir=None
    )
    if sagemaker_stuff.config.MODEL_IS_CHECKPOINT:
        checkpoint_path = os.path.join(os.path.dirname(
            __file__), f"lazy_model/{sagemaker_stuff.config.MODEL_NAME}.pth")
        model_mymodel.load_checkpoint(checkpoint_path, eval_only=True)
    else:
        raise NotImplementedError(
            "non-checkpoint model loading isn't implemented yet")
    return model_mymodel.diagnose_region


def do_analysis_job_preprocessing(dataset, callback):
    job = WeightRatioAnalysisJob(
        dataset=dataset,
        region_prediction_callback=callback,
        region_predictions_filename=os.path.join(
            sagemaker_stuff.config.DIR_OUTPUT, "region_predictions_aggregated.json")
    )
    job.get_region_predictions_from_model(
        loadingbars=sagemaker_stuff.config.LOADING_BARS,
        save_frequency=sagemaker_stuff.config.GET_REGION_PREDICTIONS_SAVE_FREQUENCY,
        tensor_size=sagemaker_stuff.config.BATCH_SIZE,
        tensor_device=torch.device(sagemaker_stuff.config.DATA_AND_MODEL_DEVICE))
    return job


def save_labels_from_dataset(dataset):
    if not sagemaker_stuff.config.DATASET_SAVE_LABELS:
        return
    labels = {}
    for filepath in dataset._filepaths:
        filepath_key = os.path.basename(
            filepath) if sagemaker_stuff.config.DATASET_SAVE_LABELS_IMAGE_BASENAME else filepath
        labels[filepath_key] = dataset.get_label(filepath)
    labels_path = os.path.join(sagemaker_stuff.config.DIR_OUTPUT,
                               sagemaker_stuff.config.DATASET_SAVE_LABELS_FILENAME)
    with open_file(labels_path, mode='w') as f:
        json.dump(labels, f)


def score_region(scorer, region, optional_appendage=None):
    region_tile = Tile(
        image=PIL.Image.fromarray(region.numpy()),
        coords=(0, 0)
    )
    if region_tile.has_enough_tissue():
        region_score = scorer(region_tile)
        if optional_appendage is not None:
            return (region_score, optional_appendage)
        else:
            return region_score
    else:
        return None


def do_tile_scoring(dataset):
    # initialize the data store
    #manager = multiprocessing.Manager()
    #scores = manager.dict()
    # pool the processes
    #pool = Pool()
    # for i, _ in loadingbar(enumerate(pool.map(
    #    func=get_best_n_scores,
    #    iterable=dataset._filepaths
    # )), total=len(dataset._filepaths)):
    #    print(f"{i} processes have been completed")
    #scoring_processes = []
    # for process_num, (filepath, label, region_generator) in enumerate(dataset.iterate_by_file()):
    #    p = Process(target=get_best_n_scores, args=(
    #        scores, filepath, region_generator))
    #    scoring_processes.append(p)
    #    p.start()
    #    print(f"Kicked off process {process_num} for {filepath}", end='')
    # wait for all processes to finish
    # while True:
    #    processes_left = len(
    #        [None for p in scoring_processes if p.exitcode is None])
    #    print(f"{processes_left} processes left...    ", end='')
    #    if processes_left == 0:
    #        time.sleep(10)
    #        print("\nDone!")
    #        break
    #    else:
    #        print(
    #            f"waiting another {sagemaker_stuff.config.UPDATE_INTERVAL} seconds")
    #        time.sleep(sagemaker_stuff.config.UPDATE_INTERVAL)
    # record the process's exist codes
    # exit_codes = [
    #    p.exitcode for p in scoring_processes]
    scorer = NucleiScorer()
    scores = {}
    for (filepath, label, region_dataset) in loadingbar(dataset.iterate_by_file(as_pytorch_datasets=True), total=len(dataset._filepaths)):
        dataloader = torch.utils.data.DataLoader(
            region_dataset,
            batch_size=sagemaker_stuff.config.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=multiprocessing.cpu_count()
        )
        this_file_scores = []
        for batch_num, batch in loadingbar(enumerate(dataloader), total=len(dataloader), leave=False, mininterval=sagemaker_stuff.config.UPDATE_INTERVAL):
            pool = multiprocessing.Pool()
            this_file_scores_unclean = pool.starmap(
                score_region,
                [(scorer, batch[region_num], batch_num*sagemaker_stuff.config.BATCH_SIZE+region_num)
                 for region_num in range(batch.shape[0])]
            )
            this_file_scores += [
                score_and_index for score_and_index in this_file_scores_unclean if score_and_index is not None]
            # for region_num in range(batch.shape[0]):
            #    region = batch[region_num]
            #    region_score = score_region(scorer, region)
            #    if region_score is not None:
            #        this_file_scores.append(
            #            (region_score, batch_num*sagemaker_stuff.config.BATCH_SIZE+region_num))
            if sagemaker_stuff.config.IS_TESTING_LOCALLY and len(this_file_scores) > 0:
                break
        this_file_scores.sort(reverse=True)
        scores[filepath] = this_file_scores
    # write data to file
    scoring_data_filepath = os.path.join(
        sagemaker_stuff.config.DIR_OUTPUT, "scoring_data.json")
    with open_file(scoring_data_filepath, 'w') as f:
        json.dump(scores, f)
