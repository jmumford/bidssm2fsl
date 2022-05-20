import numpy as np
import copy
from bids.variables.io import _get_nvols
from bids.layout.writing import build_path


def get_scan_length(img_f=None, scan_length=None, tr=None):
    if img_f is None and scan_length is None:
        raise ValueError(['Neither image nor scan_length have been specified'])
    try:
        nvols = _get_nvols(img_f)
    except Exception as e:
        if scan_length is not None:
            nvols = int(np.rint(scan_length / tr))
        else:
            msg = (
                "Unable to extract scan duration from one or more "
                "BOLD runs, and no scan_length argument was provided "
                "as a fallback. Please check that the image files are "
                "available, or manually specify the scan duration."
            )
            raise ValueError(msg) from e
    return nvols


def prep_lev1_output_dir(trans_out, output_root):
    entities_keep = ('subject', 'session', 'run', 'task')
    entities = trans_out.entities.copy()
    entities = {k: entities[k] for k in entities.keys() & entities_keep}
    pattern = ['sub-{subject}/'
               'sub-{subject}[_ses-{session}][_task-{task}][_run-{run}]']
    path_mid = build_path(entities, pattern, strict=False)
    feat_dir = output_root / 'analysis_output' / path_mid
    pattern_fsf = [
        'sub-{subject}/'
        'sub-{subject}[_ses-{session}][_task-{task}][_run-{run}]'
    ]
    path_mid_fsf = build_path(entities, pattern_fsf, strict=True)
    fsf_file = output_root / 'analysis_files' / f'{path_mid_fsf}.fsf'
    parent_dir = feat_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    return str(feat_dir), fsf_file


def get_img_filepaths(layout, collection):
    entities_keep = [
        'extension', 'suffix', 'task', 'subject', 'datatype', 'space', 'desc'
    ]
    entities = {
        key: value for key, value in collection.entities.items() if
        key in entities_keep
    }
    image_file = layout.get(return_type='object', scope='all',  **entities)

    entities_t1 = {
        key: value for key, value in entities.items() if
        key in ['subject', 'space', 'extension']
    }
    entities_t1['suffix'] = 'T1w'
    entities_t1['datatype'] = 'anat'
    anat_file = layout.get(return_type='object', scope='all', **entities_t1)
    return image_file[0], anat_file[0]


def make_con_vec(weight_row, condition_list, ev_names):
    swap_dict = {}
    for i, cond_val in enumerate(condition_list):
        swap_dict[cond_val] = weight_row[i]
    t_con = [swap_dict.get(x, 0) for x in ev_names]
    return t_con


def set_hrf(hrf_type):
    hrf_map = {
        "none": 0,
        "gamma": 2,
        "double_gamma":3
    }
    return hrf_map[hrf_type]


def make_con_mat(contrast_info, ev_names, cope_pattern):
    t_con_mat = []
    con_names = []
    f_cons_ind = []
    contrast_name_decoder = {}

    for con_num, contrast in enumerate(contrast_info):
        weight_array = np.array(contrast['weights'])
        if contrast['test'] == 't':
            contrast_num = con_num + 1
            contrast_name_decoder[contrast['name']] = [
                cope_pattern[0].replace('_copenum', str(contrast_num))
            ]
            t_con = make_con_vec(weight_array, contrast['condition_list'], ev_names)
            t_con_mat.append(t_con)
            con_names.append(contrast['name'])

        if contrast['test'] == 'F':
            contrast_name_decoder[contrast['name']] = 'ftest'
            ind_t = []
            for con_row in weight_array:
                t_con = make_con_vec(con_row, contrast['condition_list'], ev_names)
                if t_con in t_con_mat:
                    ind_t.append(t_con_mat.index(t_con))
                else:
                    t_con_mat.append(t_con)
                    ind_t.append(t_con_mat.index(t_con))
            f_cons_ind.append(ind_t)
    return t_con_mat, con_names, f_cons_ind, contrast_name_decoder


def make_lev1_fsf(
    node, layout, collection, trans_out, output_root, regs_categorized,
    reg_paths, hrf_type, add_derivatives, smoothing
):
    ev_names = regs_categorized['convolve_regs'] + regs_categorized['no_convolve_regs']
    tr = collection.entities['RepetitionTime']

    hrf = set_hrf(hrf_type)

    derivative_setting = 1 if add_derivatives else 0

    feat_dir, fsf_file = prep_lev1_output_dir(trans_out, output_root)
    image_file, anat_file = get_img_filepaths(layout, collection)

    scan_length = get_scan_length(img_f=image_file, tr=tr)

    num_evs = len(reg_paths['convolve_regs']) + len(reg_paths['no_convolve_regs'])
    if add_derivatives:
        num_evs_real = num_evs * 2
    else:
        num_evs_real = num_evs

    if node.model['options']['high_pass_filter_cutoff_hz']:
        hp_cutoff_s = 1 / node.model['options']['high_pass_filter_cutoff_hz']
    else:
        print('No highpass filter cutoff specified, setting to .01Hz (100s)')
        hp_cutoff_s = 100

    contrast_info = copy.deepcopy(node.contrasts)
    # Add Dummies to beginning
    if node.dummy_contrasts:
        if 'conditions' not in node.dummy_contrasts:
            node.dummy_contrasts['conditions'] = ev_names
        for condition in node.dummy_contrasts['conditions']:
            new_entry = {
                'name': condition,
                'weights': [1],
                'condition_list': [condition],
                'test': 't'
            }
            contrast_info.insert(0, new_entry)

    cope_pattern_lev1 = [f"{output_root}/analysis_output/"
               'sub-{subject}[/ses-{session}]/'
               'sub-{subject}[_ses-{session}][_task-{task}][_run-{run}].feat/'
               'stats/cope_copenum.nii.gz']
    t_con_mat, con_names, f_cons_ind, contrast_name_decoder = \
        make_con_mat(contrast_info, ev_names, cope_pattern_lev1)

    if derivative_setting == 1:
        t_con_mat_real = []
        for con in t_con_mat:
            con_new = [0] * num_evs * 2
            con_new[0::2] = con
            t_con_mat_real.append(con_new)
    else:
        t_con_mat_real = t_con_mat

    fsf_chunk_lev1 = {
        'set fmri(level)': 1,
        'set fmri(analysis)': 7,
        'set fmri(outputdir)': feat_dir,
        'set fmri(tr)': tr,
        'set fmri(npts)': scan_length,
        'set fmri(ndelete)': 0,
        'set fmri(multiple)': 1,
        'set fmri(filtering_yn)': 1,
        'set fmri(smooth)': smoothing,
        'set fmri(temphp_yn)': 1,
        'set fmri(mixed_yn)': 2,
        'set fmri(evs_orig)': num_evs,
        'set fmri(evs_real)': num_evs_real,
        'set fmri(ncon_orig)': len(t_con_mat),
        'set fmri(ncon_real)': len(t_con_mat),
        'set fmri(nftests_orig)': len(f_cons_ind),
        'set fmri(nftests_real)': len(f_cons_ind),
        'set fmri(thresh)': 0,
        'set fmri(regstandard)': anat_file.path.split('.nii.gz')[0],
        'set fmri(paradigm_hp)': hp_cutoff_s,
        'set fmri(ncopeinputs)': 0,
        'set fmri(confoundevs)': 1,
        'set confoundev_files(1)': reg_paths['confound_regs'],
        'set feat_files(1)': image_file.path,
        'set fmri(con_mode)': 'orig',
        'set fmri(con_mode_old)': 'orig'
    }

    conv_ev_dict = {}
    for ev_num, ev_name in enumerate(regs_categorized['convolve_regs']):
        ev_num_plus1 = ev_num + 1
        conv_ev_dict.update({
            f'set fmri(evtitle{ev_num_plus1})': ev_name,
            f'set fmri(shape{ev_num_plus1})': 3,
            f'set fmri(convolve{ev_num_plus1})': hrf,
            f'set fmri(convolve_phase{ev_num_plus1})': 0,
            f'set fmri(tempfilt_yn{ev_num_plus1})': 1,
            f'set fmri(deriv_yn{ev_num_plus1})': derivative_setting,
            f'set fmri(custom{ev_num_plus1})': str(reg_paths['convolve_regs'][ev_name]),
        })
        conv_ev_dict.update({f'set fmri(ortho{ev_num_plus1}.{idx_ev})': 0 for idx_ev in range((num_evs+1))})

    num_conv_regs = len(regs_categorized['convolve_regs'])
    noconv_ev_dict = {}
    for ev_num, ev_name in enumerate(regs_categorized['no_convolve_regs']):
        ev_num_plus1 = ev_num + 1 + num_conv_regs
        noconv_ev_dict.update({
            f'set fmri(evtitle{ev_num_plus1})': ev_name,
            f'set fmri(shape{ev_num_plus1})': 1,
            f'set fmri(convolve{ev_num_plus1})': 0,
            f'set fmri(convolve_phase{ev_num_plus1})': 0,
            f'set fmri(tempfilt_yn{ev_num_plus1})': 1,
            f'set fmri(deriv_yn{ev_num_plus1})': derivative_setting,
            f'set fmri(custom{ev_num_plus1})': str(reg_paths['no_convolve_regs'][ev_name])
        })
        noconv_ev_dict.update({
            f'set fmri(ortho{ev_num_plus1}.{idx_ev})': 0
            for idx_ev in range((num_evs+1))
        })

    real_contrast_dict = {}

    for con_num, con_name in enumerate(con_names):
        con_vec = t_con_mat_real[con_num]
        n_convals = len(con_vec)
        con_num_plus1 = con_num + 1
        real_contrast_dict[f"set fmri(conpic_real.{con_num_plus1})"] = 1
        real_contrast_dict[f"set fmri(conname_real.{con_num_plus1})"] = con_name
        real_contrast_dict.update({
            f'set fmri(con_real{con_num_plus1}.{idx_con})':
            con_vec[(idx_con - 1)] for idx_con in range(1, (n_convals+1))
        })
        if f_cons_ind:
            for idx, f_ind in enumerate(f_cons_ind):
                f_ind_real = [2*val for val in f_ind]
                fconval = 1 if 2*con_num in f_ind_real else 0
                real_contrast_dict.update({
                    f'set fmri(ftest_real{idx + 1}.{con_num_plus1})': fconval
                })
                
    orig_contrast_dict = {}
    for con_num, con_name in enumerate(con_names):
        con_vec = t_con_mat[con_num]
        n_convals = len(con_vec)
        con_num_plus1 = con_num + 1
        orig_contrast_dict[f"set fmri(conpic_orig.{con_num_plus1})"] = 1
        orig_contrast_dict[f"set fmri(conname_orig.{con_num_plus1})"] = con_name
        orig_contrast_dict.update({
            f'set fmri(con_orig{con_num_plus1}.{idx_con})':
            con_vec[(idx_con - 1)] for idx_con in range(1, (n_convals+1))
        })
        if f_cons_ind:
            for idx, f_ind in enumerate(f_cons_ind):
                fconval = 1 if con_num in f_ind else 0
                orig_contrast_dict.update({
                    f'set fmri(ftest_orig{idx + 1}.{con_num_plus1})': fconval
                })

    # I'm not including contrast masking (so setting to all 0s)
    tot_tests = len(t_con_mat) + len(f_cons_ind)
    contrast_masking = {}
    for i in range(1, tot_tests + 1):
        for j in range(1, tot_tests+1):
            if i != j:
                contrast_masking.update({f'set fmri(conmask{i}_{j})': 0})
    contrast_masking.update({'set fmri(conmask1_1)': 0})

    fsf_stub = {
        'set fmri(version)':  6.00,
        'set fmri(inmelodic)':  0,
        'set fmri(relative_yn)':  0,
        'set fmri(help_yn)':  0,
        'set fmri(featwatcher_yn)':  0,
        'set fmri(sscleanup_yn)':  0,
        'set fmri(tagfirst)':  1,
        'set fmri(inputtype)':  2,
        'set fmri(brain_thresh)':  10,
        'set fmri(critical_z)':  5.3,
        'set fmri(noise)':  0.66,
        'set fmri(noisear)':  0.34,
        'set fmri(mc)':  0,
        'set fmri(sh_yn)':  0,
        'set fmri(regunwarp_yn)':  0,
        'set fmri(gdc)':  "",
        'set fmri(dwell)':  0,
        'set fmri(te)':  35,
        'set fmri(signallossthresh)':  10,
        'set fmri(unwarp_dir)':  'y-',
        'set fmri(st)':  0,
        'set fmri(st_file)':  "",
        'set fmri(bet_yn)':  1,
        'set fmri(norm_yn)':  0,
        'set fmri(perfsub_yn)':  0,
        'set fmri(templp_yn)':  0,
        'set fmri(melodic_yn)':  0,
        'set fmri(stats_yn)':  1,
        'set fmri(motionevs)':  0,
        'set fmri(motionevsbeta)':  "",
        'set fmri(scriptevsbeta)':  "",
        'set fmri(robust_yn)':  0,
        'set fmri(randomisePermutations)':  5000,
        'set fmri(constcol)':  0,
        'set fmri(poststats_yn)':  0,
        'set fmri(threshmask)':  "",
        'set fmri(prob_thresh)':  0.05,
        'set fmri(z_thresh)':  3.1,
        'set fmri(zdisplay)':  0,
        'set fmri(zmin)':  2,
        'set fmri(zmax)':  8,
        'set fmri(rendertype)':  1,
        'set fmri(bgimage)':  1,
        'set fmri(tsplot_yn)':  0,
        'set fmri(reginitial_highres_yn)':  0,
        'set fmri(reginitial_highres_search)':  0,
        'set fmri(reginitial_highres_dof)':  3,
        'set fmri(reghighres_yn)':  0,
        'set fmri(reghighres_search)':  0,
        'set fmri(reghighres_dof)':  3,
        'set fmri(regstandard_yn)':  1,
        'set fmri(alternateReference_yn)':  0,
        'set fmri(regstandard_search)':  0,
        'set fmri(regstandard_dof)':  3,
        'set fmri(regstandard_nonlinear_yn)':  0,
        'set fmri(regstandard_nonlinear_warpres)':  10,
        'set fmri(totalVoxels)':  22364160,
        'set fmri(conmask_zerothresh_yn)':  0,
        'set fmri(alternative_mask)':  "",
        'set fmri(init_initial_highres)':  "",
        'set fmri(init_highres)':  "",
        'set fmri(init_standard)':  "",
        'set fmri(overwrite_yn)':  0,
        'set fmri(evs_vox)':  0,
        'set fmri(prewhiten_yn)':  1,
    }

    all_dict = {
        **fsf_stub,
        **fsf_chunk_lev1,
        **conv_ev_dict,
        **noconv_ev_dict,
        **real_contrast_dict,
        **orig_contrast_dict,
        **contrast_masking
    }

    with open(fsf_file, 'w') as file:
        for key, item in all_dict.items():
            if type(item) == str and item != 'orig' and item != 'y-':
                item = f'"{item}"'
            file.write(f'{key} {item}\n')

    return contrast_name_decoder, fsf_file, feat_dir


def make_higher_lev_fsf(node_mod, contrast_decoder_prev_level, output_root): 
    model_type = node_mod['model']['type']
    if model_type == 'glm':
        mix_yn = 2
        fmri_thresh = 3
    else:
        mix_yn = 3
        fmri_thresh = 0
    
    input_contrast = node_mod['entities']['contrast'].replace(' ', '_')
    # Fix for when entities are missing (e.g. run in session level)
    set_cols = set(node_mod['metadata'].columns)
    set_entities = set(node_mod['entities'].keys())
    missing_entities = list(set_cols - set_entities)
    for missing_entity in missing_entities:
        node_mod['entities'][missing_entity] = \
            list(node_mod['metadata'][missing_entity].unique())

    level = node_mod['level']
    
    pattern_options = {'session' : [
        f"{output_root}/analysis_output/"
        'sub-{subject}/[sub-{subject}_][_ses-{session}][_task-{task}]'
        f'level-{level}_{input_contrast}'    
        ],
        'subject': [
        f"{output_root}/analysis_output/"
        'sub-{subject}/[sub-{subject}_][_task-{task}]'
        f'level-{level}_{input_contrast}'    
        ],
        'dataset': [
        f"{output_root}/analysis_output/"
        '[task-{task}]'
        f'level-{level}_{input_contrast}'   
        ]
    }
    feat_pattern = pattern_options[level]
    feat_dir = build_path(node_mod['entities'], feat_pattern, strict=False)
    fsf_file = feat_dir.replace('analysis_output', 'analysis_files') + '.fsf'

    path_copes =  build_path(
        node_mod['entities'], 
        contrast_decoder_prev_level[node_mod['entities']['contrast']], 
        strict=False
    )
       
    feat_files = {f'set feat_files({ind+1})': path for ind, path in enumerate(path_copes)}

    num_input = len(path_copes)
    desmat = node_mod['X'].copy()
    num_evs = desmat.shape[1]
    contrast_info = copy.deepcopy(node_mod['contrasts'])
    #Annoying, need to rename key
    for con in contrast_info:
        con['condition_list'] = con.pop('conditions')
    ev_names = list(desmat.columns)
    #For future key
    cope_pattern_higher_lev = [f"{output_root}/analysis_output/"
               'sub-{subject}[/ses-{session}]/'
               'sub-{subject}[_ses-{session}][_task-{task}][_run-{run}].gfeat/'
               'cope1.feat/stats/cope_copenum.nii.gz']
    t_con_mat, con_names, f_cons_ind, contrast_name_decoder = \
        make_con_mat(contrast_info, ev_names, cope_pattern_higher_lev)

    ev_dict = {}
    for ev_num in range(num_evs):
        ev_num_plus1 = ev_num + 1
        ev_name = ev_names[ev_num]
        ev_dict.update({
            f'set fmri(evtitle{ev_num_plus1})': ev_name,
            f'set fmri(shape{ev_num_plus1})': 2,
            f'set fmri(convolve{ev_num_plus1})': 0,
            f'set fmri(convolve_phase{ev_num_plus1})': 0,
            f'set fmri(tempfilt_yn{ev_num_plus1})': 0,
            f'set fmri(deriv_yn{ev_num_plus1})': 0,
            f'set fmri(custom{ev_num_plus1})': "dummy"
        })
        for ev_row in range(desmat.shape[0]):
            ev_row_plus1 = ev_row + 1
            ev_dict.update({
               f'set fmri(evg{ev_row_plus1}.{ev_num_plus1})': desmat[ev_name][ev_row], 
            })
        ev_dict.update({f'set fmri(ortho{ev_num_plus1}.{idx_ev})': 0 
            for idx_ev in range((num_evs+1))})

    ev_dict.update({
        f'set fmri(groupmem.{ev_row+1})': 1 for ev_row in range(desmat.shape[0])
    })

    real_contrast_dict = {}
    contrast_name_decoder['cope_pattern'] = f"{feat_dir}.gfeat/cope1.feat/stats/cope"
    for con_num, con_name in enumerate(con_names):
        con_vec = t_con_mat[con_num]
        n_convals = len(con_vec)
        con_num_plus1 = con_num + 1
        real_contrast_dict[f"set fmri(conpic_real.{con_num_plus1})"] = 1
        real_contrast_dict[f"set fmri(conname_real.{con_num_plus1})"] = con_name
        real_contrast_dict.update({
            f'set fmri(con_real{con_num_plus1}.{idx_con})':
            con_vec[(idx_con - 1)] for idx_con in range(1, (n_convals+1))
        })
        if f_cons_ind:
            for idx, f_ind in enumerate(f_cons_ind):
                f_ind_real = [2*val for val in f_ind]
                fconval = 1 if 2*con_num in f_ind_real else 0
                real_contrast_dict.update({
                    f'set fmri(ftest_real{idx + 1}.{con_num_plus1})': fconval
                })

    tot_tests = len(t_con_mat) + len(f_cons_ind)
    contrast_masking = {}
    for i in range(1, tot_tests + 1):
        for j in range(1, tot_tests+1):
            if i != j:
                contrast_masking.update({f'set fmri(conmask{i}_{j})': 0})
    contrast_masking.update({'set fmri(conmask1_1)': 0})

    fsf_chunk = {
        'set fmri(level)': 2,
        'set fmri(analysis)': 2,
        'set fmri(outputdir)': feat_dir,
        'set fmri(tr)': 3,
        'set fmri(npts)': num_input,
        'set fmri(ndelete)': 0,
        'set fmri(multiple)': num_input,
        'set fmri(filtering_yn)': 0,
        'set fmri(smooth)': 5,
        'set fmri(temphp_yn)': 1,
        'set fmri(mixed_yn)': mix_yn,
        'set fmri(evs_orig)': num_evs,
        'set fmri(evs_real)': num_evs,
        'set fmri(ncon_orig)': len(t_con_mat),
        'set fmri(ncon_real)': len(t_con_mat),
        'set fmri(nftests_orig)': len(f_cons_ind),
        'set fmri(nftests_real)': len(f_cons_ind),
        'set fmri(thresh)': fmri_thresh,
        'set fmri(regstandard)': '',
        'set fmri(paradigm_hp)': 100,
        'set fmri(ncopeinputs)': 0,
        'set fmri(confoundevs)': 0,
        'set fmri(con_mode)': 'real',
        'set fmri(con_mode_old)': 'real'
    }
    
    fsf_stub = {
        'set fmri(version)':  6.00,
        'set fmri(inmelodic)':  0,
        'set fmri(relative_yn)':  0,
        'set fmri(help_yn)':  0,
        'set fmri(featwatcher_yn)':  0,
        'set fmri(sscleanup_yn)':  0,
        'set fmri(tagfirst)':  1,
        'set fmri(inputtype)':  2,
        'set fmri(brain_thresh)':  10,
        'set fmri(critical_z)':  5.3,
        'set fmri(noise)':  0.66,
        'set fmri(noisear)':  0.34,
        'set fmri(mc)':  0,
        'set fmri(sh_yn)':  0,
        'set fmri(regunwarp_yn)':  0,
        'set fmri(gdc)':  "",
        'set fmri(dwell)':  0,
        'set fmri(te)':  35,
        'set fmri(signallossthresh)':  10,
        'set fmri(unwarp_dir)':  'y-',
        'set fmri(st)':  0,
        'set fmri(st_file)':  "",
        'set fmri(bet_yn)':  1,
        'set fmri(norm_yn)':  0,
        'set fmri(perfsub_yn)':  0,
        'set fmri(templp_yn)':  0,
        'set fmri(melodic_yn)':  0,
        'set fmri(stats_yn)':  1,
        'set fmri(motionevs)':  0,
        'set fmri(motionevsbeta)':  "",
        'set fmri(scriptevsbeta)':  "",
        'set fmri(robust_yn)':  0,
        'set fmri(randomisePermutations)':  5000,
        'set fmri(constcol)':  0,
        'set fmri(poststats_yn)':  1,
        'set fmri(threshmask)':  "",
        'set fmri(prob_thresh)':  0.05,
        'set fmri(z_thresh)':  3.1,
        'set fmri(zdisplay)':  0,
        'set fmri(zmin)':  2,
        'set fmri(zmax)':  8,
        'set fmri(rendertype)':  1,
        'set fmri(bgimage)':  1,
        'set fmri(tsplot_yn)':  0,
        'set fmri(reginitial_highres_yn)':  0,
        'set fmri(reginitial_highres_search)':  0,
        'set fmri(reginitial_highres_dof)':  3,
        'set fmri(reghighres_yn)':  0,
        'set fmri(reghighres_search)':  0,
        'set fmri(reghighres_dof)':  3,
        'set fmri(regstandard_yn)':  1,
        'set fmri(alternateReference_yn)':  0,
        'set fmri(regstandard_search)':  0,
        'set fmri(regstandard_dof)':  3,
        'set fmri(regstandard_nonlinear_yn)':  0,
        'set fmri(regstandard_nonlinear_warpres)':  10,
        'set fmri(totalVoxels)':  22364160,
        'set fmri(conmask_zerothresh_yn)':  0,
        'set fmri(alternative_mask)':  "",
        'set fmri(init_initial_highres)':  "",
        'set fmri(init_highres)':  "",
        'set fmri(init_standard)':  "",
        'set fmri(overwrite_yn)':  0,
        'set fmri(evs_vox)':  0,
        'set fmri(prewhiten_yn)':  1,
    }
    
    all_dict = {
        **fsf_stub,
        **feat_files,
        **fsf_chunk,
        **ev_dict,
        **real_contrast_dict,
        **contrast_masking
    }

    with open(fsf_file, 'w') as file:
        for key, item in all_dict.items():
            if type(item) == str and item != 'orig' and item != 'y-' and item != 'real':
                item = f'"{item}"'
            file.write(f'{key} {item}\n')

    return contrast_name_decoder, fsf_file