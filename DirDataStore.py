# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:19:23 2019

@author: sdhruba
"""
##############################################
# Return list of filepaths and assigned labels...
import os
import sys
import numpy as np

def DirDataStore(dir, filter = False, filter_dict = [ ]):
    subdirs = os.listdir(dir)                                                   # Subdirectories
    if filter:                                                                  # Filter subdirectory names
        if filter_dict:                                                         
            subdir_idx = [any([c == s.lower() for c in filter_dict]) for s in subdirs]
            subdirs    = np.array(subdirs)[np.array(subdir_idx)].tolist()
        else:
            print('Error: filter_dict cannot be empty if filter = True!\n')
            sys.exit(1)
    else:
        filter_dict = [s.lower() for s in subdirs]                             # Use subdirectory names as filters
    
    
    # List of all input filepaths & labels...
    FileRoot, FileLabel = [ ], [ ]
    for sdir in  subdirs:
#        print(sdir)
        match = [c in sdir.lower() for c in filter_dict]
        match = [c == sdir.lower() for c in filter_dict] if np.sum(match) > 1 else match    # #match > 1 [i.e., GunGym]
        label = filter_dict[int(np.nonzero(match)[0])]
        
        # Tree structure...
        roots, dirs, files = [ ], [ ], [ ]
        for r, d, f in os.walk(os.path.join(dir, sdir)):
            roots.append(r);    dirs.append(d);    files.append(f)
        nefiles_idx = [not(not f) for f in files]                              # Non-empty files
        roots, files = np.array(roots)[np.array(nefiles_idx)].tolist(), np.array(files)[np.array(nefiles_idx)].tolist()
        for i in range(len(roots)):
            for j in range(len(files[i])):
                FileRoot.append(os.path.join(roots[i], files[i][j]))
                FileLabel.append(label)
    
    # Create datastore object to return...
    DSobj = [FileRoot, FileLabel]                                               # Return object
    
    return DSobj