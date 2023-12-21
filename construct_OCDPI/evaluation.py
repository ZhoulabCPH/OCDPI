import random
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from model import GDL
from utils.dataset import OV
from utils.util import make_batch, collate, CoxLoss, weight_init
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

survminer = importr('survminer')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluation(model, cohort='', cutpoint=None):
    workspace = pd.read_csv(f'../datasets/clinical_data/{cohort}.csv')
    nodes_path = f'../datasets/graphs/{cohort}_nodes.npy'
    edges_path = f'../datasets/graphs/{cohort}_edges.npy'

    data = OV(workspace, nodes_path, edges_path)
    data_loader = DataLoader(data, 8, shuffle=False, num_workers=0, drop_last=False,
                                  collate_fn=collate)
    report = pd.DataFrame()
    slides_name = []
    OCDPIs = np.array([])
    OS_times = np.array([])
    OSs = np.array([])

    with torch.no_grad():
        for step, graphs in enumerate(data_loader):
            batch_graphs, batch_OSs, batch_OS_times, batch_slides_name = make_batch(graphs)
            batch_OCDPI, _ = model.forward(batch_graphs)
            OCDPIs = np.append(OCDPIs, batch_OCDPI.detach().cpu().numpy())
            slides_name = slides_name + batch_slides_name
            OS_times = np.append(OS_times, batch_OS_times)
            OSs = np.append(OSs, batch_OSs.detach().cpu().numpy())

    report['slides'] = list(slides_name)
    report['OS.time'] = list(OS_times)
    report['OS'] = list(OSs)
    report['OCDPI'] = list(OCDPIs)
    pandas2ri.activate()
    r_dataframe = pandas2ri.py2rpy_pandasdataframe(report)
    if cohort == 'TCGA_discovery':
        cutpoint = survminer.surv_cutpoint(r_dataframe, time="OS.time", event="OS",
                                           variables='OCDPI')
        cutpoint = cutpoint.rx2['cutpoint'][0]

    GROUP = [1 if index > cutpoint else 0 for index in report.loc[:,'OCDPI'].to_list()]
    report['GROUP'] = GROUP
    cph = CoxPHFitter()

    cph.fit(report.loc[:, ['OS.time', 'OS', 'GROUP']], duration_col='os', event_col='os_state')
    HR = cph.summary['exp(coef)'].values

    report_low_OCDPI = report[report.loc[:, 'GROUP_survpoint'] == 0]
    report_high_OCDPI = report[report.loc[:, 'GROUP_survpoint'] == 1]
    log_rank_p = logrank_test(report_low_OCDPI.loc[:, 'OS.time'].to_list(), report_high_OCDPI.loc[:, 'OS.time'].to_list(),
                               event_observed_A=report_low_OCDPI.loc[:, 'OS'].to_list(),
                               event_observed_B=report_high_OCDPI.loc[:, 'OS'].to_list())

    log_rank_p = log_rank_p.p_value
    return report, HR, log_rank_p, cutpoint


if __name__ == '__main__':
    model = GDL().cuda()
    ckpt = torch.load(
        f'../checkpoints/checkpoint_GDL.pth',
        map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    model.eval()
    report_TCGA_discovery, HR_TCGA_discovery, log_rank_p_TCGA_discovery, cutpoint_TCGA_discovery = evaluation(model,
                                                                                                              cohort='TCGA_discovery',
                                                                                                              cutpoint=None)
    report_PLCO, HR_PLCO, log_rank_p_PLCO, cutpoint_PLCO = evaluation(model, cohort='PLCO',
                                                                      cutpoint=cutpoint_TCGA_discovery)
    report_HMUCH, HR_HMUCH, log_rank_p_HMUCH, cutpoint_HMUCH = evaluation(model,
                                                                          cohort='HMUCH',
                                                                          cutpoint=cutpoint_TCGA_discovery)













