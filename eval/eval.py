import torch
import os
import sys

import numpy as np


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..utils import tqdm, HiddenPrints



class Evaluator():
    """ 
    Evaluation class. 
    Performs evalution on a given model and DataHandler. 

    Two main functions can be used: 
        - eval: which runs evaluation on the DataHandler query set.
        - eval_all: which use DataHandler instantiated with eval_all=True. 
                    It runs eval on all the loaders provided by the DataHandler.  
    
    Arguments:
        - model: pytorch model to be evaluated.
        - cfg: cfg object of the model.
        - data_handler: DataHandler object of the dataset.
    """
    def __init__(self,
                 model,
                 cfg,
                 data_handler,
                 output_folder='./Experiments_FSFCOS/Results'):
        self.model = model
        self.cfg = cfg

        self.data_handler = data_handler
        self.device = cfg.MODEL.DEVICE
        self.output_folder = output_folder

        self.categories =  None
        self.current_classes = None

    def eval(self, verbose=True, all_classes=True, verbose_classes=True, loaders=None, seed=None):
        """
        Eval function on a single data loader (or couple query/support loaders)

        Arguments:
            - verbose: print eval results at the end of computation.
            - all_classes:  
        """
        if seed is not None:
            self.data_handler.rng_handler.update_seeds(seed)

        if self.cfg.FEWSHOT.ENABLED:
            if loaders is None:
                query_loader, support_loader, classes = self.data_handler.get_dataloader(
                    seed=seed
                )
            else:
                query_loader, support_loader, classes = loaders

            self.current_classes = classes
            if verbose_classes:
                print('Evaluation on classes: {}'.format(str(classes)))

            self.categories = {
                idx: v['name']
                for idx, v in query_loader.dataset.coco.cats.items()
            }

            self.contiguous_label_map = query_loader.dataset.contiguous_category_id_to_json_id
            predictions = self.compute_pred_fs(query_loader, support_loader,
                                            classes)
        else:
            query_loader = self.data_handler.get_dataloader(seed=seed)
            classes = np.array(list(query_loader.dataset.coco.cats.keys())) + 1
            predictions = self.compute_pred(query_loader)



        if self.has_pred(predictions):

            for pred in predictions:
                pred.add_field("objectness",
                                torch.ones(len(pred), device=self.device))

            # dirty hack to remove prints from pycocotools
            with HiddenPrints():

                coco_results = self.prepare_for_coco_detection(predictions,
                                                    query_loader.dataset)

                res = self.evaluate_predictions_on_coco(
                    query_loader.dataset.coco, coco_results,
                    os.path.join(self.output_folder, 'res.json'),
                    'bbox',
                    classes=classes)

                res_per_class = {}
                for c in classes:
                    res_per_class[c] = self.evaluate_predictions_on_coco(query_loader.dataset.coco,
                                                            coco_results,
                                                            os.path.join(
                                                                self.output_folder,
                                                                'res.json'),
                                                            'bbox',
                                                            classes=[c])
            if verbose:
                self.eval_summary(res, res_per_class, all_classes=all_classes)
            return res, res_per_class

        else:
            return {}, {}

    def eval_all(self, n_episode=1, verbose=True, seed=None):
        """
        Similar to eval function except it loop over the multiple dataloaders returned 
        by the DataHandler. (DataHandler must have eval_all=True).

        Results are then accumulated and stored in a pandas dataframe. 
        
        """
        assert self.data_handler.eval_all == True, 'Use eval_all with eval_all=True in DataHandler'
        accumulated_res_test = {}
        accumulated_res_train = {}
        all_res = {
            'train': accumulated_res_test,
            'test': accumulated_res_train
        }

        for eval_ep in range(n_episode):
            if seed is not None:
                self.data_handler.rng_handler.update_seeds(seed)
            loaders = self.data_handler.get_dataloader(seed=seed)
            for setup in ['train', 'test']:
                res_all_cls = {}
                for q_s_loaders in loaders[setup]:
                    _, res_cls = self.eval(verbose=False, loaders=q_s_loaders, seed=seed)
                    # this will overwrite some keys if the last batch is padded
                    # but only one eval is retained for each class
                    res_all_cls.update(res_cls)

                for k, v in res_all_cls.items():
                    if not k in all_res[setup]:
                        all_res[setup][k] = []
                    all_res[setup][k].append(v.stats)

        for setup in ['train', 'test']:
            for k, v in all_res[setup].items():
                all_res[setup][k] = np.vstack(all_res[setup][k]).mean(axis=0)

        return self.prettify_results_fs(all_res, verbose=verbose)

    def eval_no_fs(self, seed=None, verbose=False):
        """
        Eval without fewshot.  
        """
        overall_res, res_per_class = self.eval(seed=seed, verbose=verbose)
        for k in res_per_class:
            res_per_class[k] = res_per_class[k].stats

        return self.prettify_results(res_per_class, verbose=verbose)

    def eval_summary(self, res, res_per_class, all_classes=True):
        """
        Format results.  
        """
        classes = list(res_per_class.keys())
        sep = '+{}+'.format('-'*77)
        print('''\n{}\n\t\tAll classes {:<30}\n{}'''.format(
                    sep,
                    self.get_cat_names(classes),
                    sep))
        res.summarize()
        if all_classes:
            for c, res in res_per_class.items():
                print('''\n{}\n\t\tClass {}\n{}'''.format(
                    sep,
                    '{}: '.format(c) + self.categories[c-1],
                    sep))
                res.summarize()

    def get_cat_names(self, classes):
        return ", ".join([
            '\n\t\t{}: '.format(c) + self.categories[c-1] for c in classes
        ])


    def compute_pred_fs(self, query_loader, support_loader=None, classes=None):
        """
        Model inference on a query_loader with fewshot. 
        """
        predictions = []
        with torch.no_grad():
            # for iteration, (images, targ,
            #                 img_id) in enumerate(tqdm(query_loader,
            #                                      desc='Computing predictions')):
            for iteration, (images, targ,
                            img_id) in enumerate(query_loader):
                support = self.model.compute_support_features(
                    support_loader, self.device)
                images = images.to(self.device)
                pred_batch = self.model(images, classes=classes, support=support)
                for idx, pred in enumerate(pred_batch):
                    pred.add_field('image_id', torch.tensor(
                        img_id[idx]))  # store img_id as tensor for convenience
                    predictions.append(pred.to('cpu'))

        return predictions

    def compute_pred(self, query_loader, support_loader=None, classes=None):
        """
        Model inference on a query_loader without fewshot. 
        """
        predictions = []
        with torch.no_grad():
            for iteration, (images, targ, img_id) in enumerate(query_loader):
                images = images.to(self.device)
                pred_batch = self.model(images)
                for idx, pred in enumerate(pred_batch):
                    pred.add_field('image_id', torch.tensor(
                        img_id[idx]))  # store img_id as tensor for convenience
                    predictions.append(pred.to('cpu'))

        return predictions

    def evaluate_predictions_on_coco(self,
        coco_gt, coco_results, json_result_file, iou_type="bbox",
        classes=None):
        """
        Run coco evaluation using pycocotools. 
        """
        coco_dt = coco_gt.loadRes(coco_results)
        # self.ignore_dataset_annot_without_pred(coco_gt, coco_dt, classes)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        # coco_eval.params.catIds = [c - 1 for c in classes]
        coco_eval.params.catIds = [self.contiguous_label_map[c] for c in classes]
        coco_eval.params.imgIds = list(set([
            det['image_id'] for det in list(coco_dt.anns.values())
        ]))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval

    def ignore_dataset_annot_without_pred(self, coco_gt, coco_dt, classes):
        img_with_predictions = set([det['image_id'] for det in list(coco_dt.anns.values())])
        gt_anns = coco_gt.anns
        classes_json = [self.contiguous_label_map[c] for c in classes]
        rm_keys = []
        for k, v in gt_anns.items():
            if v['image_id'] not in img_with_predictions or \
                (classes is not None and v['category_id'] not in classes_json):
                # category id is not necesarily contiguous
                gt_anns[k]['ignore'] = 1
                rm_keys.append(k)
            elif v['image_id'] not in img_with_predictions:
                del coco_gt.imgs[v['image_id']]

        for k in rm_keys:
            del gt_anns[k]

    def prepare_for_coco_detection(self, predictions, dataset):
        """
        Convert predictions from model into coco format detections. 
        """
        # assert isinstance(dataset, COCODataset)
        coco_results = []
        for prediction in predictions:
            image_id = prediction.get_field('image_id').item()
            original_id = dataset.id_to_img_map[image_id]
            if len(prediction) == 0:
                continue

            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction = prediction.resize((image_width, image_height))
            prediction = prediction.convert("xywh")

            boxes = prediction.bbox.tolist()
            scores = prediction.get_field("scores").tolist()
            labels = prediction.get_field("labels").tolist()

            mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": mapped_labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def has_pred(self, predictions):
        nb_pred = 0
        for pred in predictions:
            nb_pred += len(pred)

        return nb_pred > 0


    """
    Prettify method build pandas dataframes from results of evaluation. 
    """
    def prettify_results_fs(self, results, verbose=True):
        import pandas as pd
        metrics = {}

        metrics['Measure'] = ['AP'] * 6 + ['AR'] * 6
        metrics['IoU'] = [
            '0.50:0.95',
            '0.50',
            '0.75',
        ] + ['0.50:0.95'] * 9
        metrics['Area'] = ['all', 'all', 'all', 'small', 'medium', 'large'] * 2

        df_metrics = pd.DataFrame.from_dict(metrics)

        df_train = pd.DataFrame.from_dict(results['train'])
        df_train = df_train.reindex(sorted(df_train.columns), axis=1)
        df_test = pd.DataFrame.from_dict(results['test'])
        df_test = df_test.reindex(sorted(df_test.columns), axis=1)

        train_cls = list(results['train'].keys())
        df_all = pd.concat([df_metrics, df_train, df_test], axis=1)

        df_all = df_all.set_index(['Measure', 'IoU', 'Area'])
        columns = [('Train classes', c) if c in train_cls else
                   ('Test classes', c) for c in df_all.columns]
        df_all.columns = pd.MultiIndex.from_tuples(columns)

        if verbose:
            print(df_all)
        return df_all

    def prettify_results(self, results, verbose=True):
        import pandas as pd
        metrics = {}

        metrics['Measure'] = ['AP'] * 6 + ['AR'] * 6
        metrics['IoU'] = [
            '0.50:0.95',
            '0.50',
            '0.75',
        ] + ['0.50:0.95'] * 9
        metrics['Area'] = ['all', 'all', 'all', 'small', 'medium', 'large'] * 2

        df_metrics = pd.DataFrame.from_dict(metrics)

        df_train = pd.DataFrame.from_dict(results)
        df_train = df_train.reindex(sorted(df_train.columns), axis=1)


        train_cls = list(results.keys())
        df_all = pd.concat([df_metrics, df_train], axis=1)

        df_all = df_all.set_index(['Measure', 'IoU', 'Area'])
        columns = [('Train classes', c) for c in df_all.columns]
        df_all.columns = pd.MultiIndex.from_tuples(columns)

        if verbose:
            print(df_all)
        return df_all