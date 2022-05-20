#!/usr/bin/env python

from bids.layout import BIDSLayout
from bids.layout.writing import build_path
from bids.modeling import BIDSStatsModelsGraph
from bids.modeling import transformations as tm
import sys
from pathlib import Path
import pandas as pd
from bids.variables import DenseRunVariable
from itertools import chain
import numpy as np
import sys  
sys.path.insert(0, '/Users/jeanettemumford/Dropbox/Research/Projects/Fsf_converter/bidssm_to_fsl')
from fsf_maker import make_lev1_fsf, make_higher_lev_fsf
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import os.path as op

def snake_to_camel(string):
    words = string.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])


def trim_instructions(instructions):
    transformation_names = [val['name'] for val in instructions]
    if 'Convolve' in transformation_names:
        convolve_idx = transformation_names.index('Convolve')
        trimmed_instructions = instructions[:convolve_idx].copy()
        convolve_regs = instructions[convolve_idx]['input'].copy()
    else:
        trimmed_instructions = instructions.copy()
        convolve_regs = None
    return trimmed_instructions, convolve_regs


def prep_onset_dir(trans_out, output_root):
    entities_keep = ('subject', 'session', 'run', 'task')
    entities = trans_out.entities.copy()
    entities = {k: entities[k] for k in entities.keys() & entities_keep} 
    pattern = ['sub-{subject}[/ses-{session}]/regressor_files/' 
               'sub-{subject}[_ses-{session}][_task-{task}][_run-{run}]']
    path_mid = build_path(entities, pattern, strict=True)
    file_root = path_mid.split('/')[-1]
    desmat_dir = output_root / 'analysis_files' / path_mid.split('/sub')[0]
    desmat_dir.mkdir(parents=True, exist_ok=True)
    return desmat_dir, file_root


def categorize_regressors(node, convolve_regs, trans_out):
    """
    Three sets of regressors, main design matrix regressors involve
    those to be convolved and not convolved, but used in contrast, last set is
    confound regressors (not used in contrasts and not convolved)
    """
    # Just in case they don't use x (although they should)
    model_name = list(node.model.keys())[0]
    specified_reg_name_list = node.model.get(model_name).copy()
    specified_reg_names = set(specified_reg_name_list) - {1}
    dense_reg_options = {
        key for key, item in trans_out.variables.items() if 
        isinstance(item, DenseRunVariable)
    }

    contrast_regs = set()
    for contrast in node.contrasts:
        contrast_regs_loop = set(contrast['condition_list'])
        contrast_regs = contrast_regs | contrast_regs_loop
    
    convolve_regs = set(convolve_regs)
    confound_regs = specified_reg_names & (dense_reg_options - contrast_regs)
    main_convolve_regs = specified_reg_names & convolve_regs
    main_no_convolve_regs = specified_reg_names & (contrast_regs - convolve_regs)
    regs_missing = (specified_reg_names - 
        (confound_regs | main_convolve_regs | main_no_convolve_regs))
    contrast_reg_missing = contrast_regs - specified_reg_names

    if regs_missing:
        raise ValueError([f'X specification includes variable(s) {regs_missing}, '
                'but these were either not data matrix or were not convolved.'])
    if contrast_reg_missing:
        raise ValueError([f'Contrast specified refers to {contrast_reg_missing}, '
                'but this is not defined in the model.'])

    #put back in order so users are not confused
    confound_regs_list = [
        name for name in specified_reg_name_list if name in confound_regs
    ]
    convolve_regs_list = [
        name for name in convolve_regs if name in main_convolve_regs
    ]
    no_convolve_reg_list = [
        name for name in specified_reg_name_list if name in main_no_convolve_regs
    ]
    regs_categorized = {
        "confound_regs": confound_regs_list,
        "convolve_regs": convolve_regs_list,
        "no_convolve_regs": no_convolve_reg_list
    }
    return regs_categorized


def write_run_level_des_files(trans_out, node, convolve_regs, output_root):   
    desmat_dir, file_root = prep_onset_dir(trans_out, output_root)

    regs_categorized = categorize_regressors(node, convolve_regs, trans_out)
    reg_paths = {key: {} for key in regs_categorized}
    if regs_categorized['confound_regs']:
        confound_regmat = pd.DataFrame()
        for confound_reg in regs_categorized['confound_regs']:
            confound_regmat = pd.concat(
                [confound_regmat, trans_out.variables.get(confound_reg).values],
                axis=1
            )
        # Fix NA that can pop up in first TR. 
        idx, _ = np.where(confound_regmat.isna())
        if any(idx>0):
            raise ValueError(['Regressor from data matrix has NaNs beyond first' 
                'time point']) 

        confound_regmat = confound_regmat.fillna(0)
        confound_regmat_filepath = desmat_dir / f"{file_root}_confound_regressors.txt"
        reg_paths['confound_regs'] = confound_regmat_filepath
        confound_regmat.to_csv(
            confound_regmat_filepath, sep="\t",
            header=False, index=False
        )
    if reg_paths['confound_regs']:
        reg_paths['confound_regs'] = str(reg_paths['confound_regs'])
    else:
        reg_paths['confound_regs'] = 0

    for reg_to_convolve in regs_categorized['convolve_regs']:
        reg_info = trans_out.variables.get(reg_to_convolve).to_df(entities=False)
        sparse_3col = reg_info[['onset', 'duration', 'amplitude']]
        sparse_3col_filepath = desmat_dir / f"{file_root}_{reg_to_convolve.replace('.','_')}.txt"
        reg_paths['convolve_regs'][reg_to_convolve] = sparse_3col_filepath
        sparse_3col.to_csv(
            sparse_3col_filepath, 
            sep="\t", header=False, index=False
        )
    for reg_no_convolve in regs_categorized['no_convolve_regs']:
        reg_info = pd.DataFrame(trans_out.variables.get(reg_no_convolve).values)
        reg_info_filepath = desmat_dir / f"{file_root}_{reg_no_convolve.replace('.','_')}.txt"
        reg_paths['no_convolve_regs'][reg_no_convolve] = reg_info_filepath
        reg_info.to_csv(
            reg_info_filepath, 
            sep="\t", header=False, index=False
        )
    return reg_paths


def load_graph(graph, node=None, inputs=None, **filters):
    if node is None:
        node = graph.root_node

    specs = node.run(inputs, group_by=node.group_by, **filters)
    outputs = list(chain(*[s.contrasts for s in specs]))
    base_entities = graph.model["input"]

    all_specs = {
        node.name: [
            {
                'contrasts': [c._asdict() for c in spec.contrasts],
                'entities': {**base_entities, **spec.entities},
                'level': spec.node.level,
                'X': spec.X,
                'name': spec.node.name,
                'model': spec.node.model,
                # Metadata is only used in higher level models; save space
                'metadata': spec.metadata if spec.node.level != "run" else None,
            }
            for spec in specs
        ]
    }

    for child in node.children:
        all_specs.update(
            load_graph(graph, child.destination, outputs, **child.filter)
        )

    return all_specs


def replace_reg_dir(featdir):
    import os
    import numpy as np
    import shutil
    if not os.path.isdir(featdir):
        print('Input featdir directory does not exist')
        return
    if not os.path.exists(f'{featdir}/example_func.nii.gz'):
        print('example_func.nii.gz not found \n')
        print('Check that input is feat directory')
        return
    shutil.copy(f'{featdir}/reg/standard.nii.gz', f'{featdir}/standart.nii.gz')
    shutil.rmtree(f'{featdir}/reg')
    os.makedirs(f'{featdir}/reg')
    shutil.move(f'{featdir}/standart.nii.gz', f'{featdir}/reg/standard.nii.gz')
    # As long as my tool was used to set up fsf's, standard is subject's highres
    shutil.copy(f'{featdir}/reg/standard.nii.gz', f'{featdir}/reg/highres.nii.gz')
    ident_mat = np.identity(4)
    mat_names = [
        'example_func2highres', 'example_func2standard', 'highres2example_func',
        'highres2standard', 'standard2example_func', 'standard2highres'
    ]
    for mat_name in mat_names:
        np.savetxt(f'{featdir}/reg/{mat_name}.mat', ident_mat, delimiter=" ")



def make_reg_fixer_script(feat_dirs, output_root):
    import os, sys, stat
    code_filename = output_root / 'analysis_files/2_fix_regdirs.py'
    with open(code_filename, 'w') as file:
        file.write('#!/usr/bin/env python \n \n')
        file.write('from run_btf import replace_reg_dir \n') 
        for feat_dir in feat_dirs:
            file.write(f"replace_reg_dir('{feat_dir}') \n")
    os.chmod(code_filename, stat.S_IXUSR)


def get_parser():
    """Build parser object"""
    parser = ArgumentParser(
        prog='bidssm2fsl',
        description='bidssm2fsl: Generate FSL analysis files from BIDS Stats Model Spec',
        formatter_class=RawTextHelpFormatter,
    )

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument(
        'bids_dir',
        action='store',
        type=op.abspath,
        help='the root folder of a BIDS valid dataset (sub-XXXXX folders should '
        'be found at the top level in this folder).',
    )
    parser.add_argument(
        'fmriprep_dir',
        action='store',
        type=op.abspath,
        help='the root folder of fmriprep data',
    )
    parser.add_argument(
        'output_dir',
        action='store',
        type=op.abspath,
        help='the output path for the outcomes of preprocessing and visual reports',
    )
    parser.add_argument(
        'database_path',
        action='store',
        type=op.abspath,
        help="Path to directory containing SQLite database indices "
        "for this BIDS dataset. "
        "If a value is passed and the file already exists, "
        "indexing is skipped.",
    )
    parser.add_argument(
        'model',
        action='store',
        help='location of BIDS model description',
    )
    ##
    g_prep = parser.add_argument_group('Preprocessing settings for FSL')
    g_prep.add_argument(
        '-s',
        '--smoothing',
        default=5,
        action='store',
        help="Amount of spatial smoothing applied to data prior to model fit (FWHM mm kernel).  "
        "Default = 5mm.",
    )
    g_prep.add_argument(
        '-omit_deriv',
        '--omit_regressor_derivatives',
        action='store_false',
        help="Use this flag to omit derivatives of regressors in model.  "
        "Otherwise derivatives will be included.",
    )
    g_prep.add_argument(
        '-hrf',
        '--hrf_type',
        choices=['none', 'gamma', 'double_gamma'],
        action='store',
        default = 'double_gamma',
        help="HRF to use in convolution. Default = double_gamma.",
    )
    return parser


def main(argv=None):
    opts = get_parser().parse_args(argv)

    root = opts.bids_dir
    derivatives = opts.fmriprep_dir
    output_root = Path(opts.output_dir)
    database_path = opts.database_path
    model_path = opts.model
    hrf_type = opts.hrf_type
    add_ev_derivatives = opts.omit_regressor_derivatives
    smoothing = opts.smoothing

    layout = BIDSLayout(
            root = root,
            derivatives=derivatives,
            database_path=database_path,
            reset_database=False
    )

    graph = BIDSStatsModelsGraph(layout, model_path)    
    graph.load_collections()


    node_level_name = {node.level: node.name for node in graph.nodes.values()}
    levels = list(node_level_name.keys())

    if levels[0] != 'run':
        sys.exit(['This tool can only start with a time series analysis '
                'first node in spec must have level: run'])

    if levels.count('run') != 1:
        sys.exit(['This tool assumes exactly one run level node'])

    runlev_node = graph.nodes[node_level_name['run']]
    runlev_name = snake_to_camel(runlev_node.name.replace('-', '_'))

    run_trans = runlev_node.transformations.copy()
    apply_trans = tm.TransformerManager(run_trans['transformer'])

    trimmed_instructions, convolve_regs = trim_instructions(
            run_trans['instructions']
        )

    feat_commands = []
    feat_dirs = []
    for collection in runlev_node._collections:
        trans_out = apply_trans.transform(
            collection.clone(), trimmed_instructions
        )

        reg_paths = write_run_level_des_files(
            trans_out, runlev_node, convolve_regs, output_root
        )
        regs_categorized = categorize_regressors(runlev_node, convolve_regs, trans_out)

        contrast_name_decoder_lev1, fsf_file, feat_dir = make_lev1_fsf(
        runlev_node, layout, collection, trans_out, output_root, regs_categorized,
        reg_paths, hrf_type, add_ev_derivatives, smoothing
        )
        feat_commands.append(f'feat {fsf_file}')
        feat_dirs.append(feat_dir + '.feat')

    make_reg_fixer_script(feat_dirs, output_root)
    feat_code_file = output_root / 'analysis_files/1_run_lev1.txt'
    with open(feat_code_file, 'w') as file:
        for feat_command in feat_commands:
            file.write(feat_command + '\n')

    all_contrast_decoders = {}
    all_contrast_decoders[node_level_name['run']] = contrast_name_decoder_lev1

    all_specs = load_graph(graph, node=None, inputs=None)
    node_names = list(all_specs.keys())
    node_names = node_names[1:]
    count = 3
    for node_name in node_names:
        feat_commands = []
        node_mods = all_specs[node_name]
        for node_mod in node_mods:
            parent = graph.nodes[node_name].parents[0].source.name
            contrast_decoder = all_contrast_decoders[parent]
            if contrast_decoder[node_mod['entities']['contrast']] == 'ftest':
                print(f"Skipping contrast {node_mod['entities']['contrast']} "
                    'since it is f-test')
                continue
            contrast_name_decoder, fsf_file = \
                make_higher_lev_fsf(node_mod, contrast_decoder, output_root)
            all_contrast_decoders[node_name] = contrast_name_decoder
            feat_commands.append(f'feat {fsf_file}')
        feat_code_file = output_root / f'analysis_files/{count}_run_node_{node_name}.txt'
        count = count + 1
        with open(feat_code_file, 'w') as file:
            for feat_command in feat_commands:
                file.write(feat_command + '\n')





if __name__ == "__main__":
    main(sys.argv[1:])
