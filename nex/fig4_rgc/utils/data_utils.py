from itertools import chain
import numpy as np
import h5py
import pandas as pd


def read_data(
    start_n_scan,
    num_datapoints_per_scanfield,
    cell_id,
    rec_ids,
    noise_name,
    path_prefix=".",
):
    # Only loaded for visualization.
    file = h5py.File(f"{path_prefix}/data/{noise_name}.h5", 'r+')
    noise_stimulus = file["k"][()]
    noise_stimulus = noise_stimulus[:, :, start_n_scan:start_n_scan+num_datapoints_per_scanfield]
    noise_full = np.concatenate([noise_stimulus for _ in range(len(rec_ids))], axis=2)
    
    setup = pd.read_pickle(f"{path_prefix}/results/data/setup.pkl")
    recording_meta = pd.read_pickle(f"{path_prefix}/results/data/recording_meta.pkl")
    stimuli_meta = pd.read_pickle(f"{path_prefix}/results/data/stimuli_meta_{cell_id}.pkl")
    labels_df = pd.read_pickle(f"{path_prefix}/results/data/labels_lowpass_{cell_id}.pkl")
    
    # TODO Change to file that contains all outputs.
    bc_output = pd.read_pickle(f"{path_prefix}/results/data/off_bc_output_{cell_id}.pkl")
    
    setup = setup[setup["cell_id"] == cell_id]
    setup = setup[setup["rec_id"].isin(rec_ids)]
    
    stimuli_meta = stimuli_meta[stimuli_meta["cell_id"] == cell_id]
    
    bc_output = bc_output[bc_output["cell_id"] == cell_id]
    bc_output = bc_output[bc_output["rec_id"].isin(rec_ids)]
    
    recording_meta = recording_meta[recording_meta["cell_id"] == cell_id]
    recording_meta = recording_meta[recording_meta["rec_id"].isin(rec_ids)]
    
    labels_df = labels_df[labels_df["cell_id"] == cell_id]
    labels_df = labels_df[labels_df["rec_id"].isin(rec_ids)]
    
    # Contrain the number of labels.
    constrained_ca_activities = np.stack(labels_df["ca"].to_numpy())[:, start_n_scan:start_n_scan+num_datapoints_per_scanfield].tolist()
    labels_df["ca"] = constrained_ca_activities

    ### If we use multiple labels per image, we have to interpolate the "images" (i.e. the BC acitvities) here
    constrained_activities = np.stack(bc_output["activity"].to_numpy())[:, start_n_scan:start_n_scan+num_datapoints_per_scanfield].tolist()
    bc_output["activity"] = constrained_activities

    ### Concatenate the BC activity along the recording ids
    # Contrain the number of stimulus images.
    bc_output_concatenated = bc_output.groupby("bc_id", sort=False)["activity"].apply(lambda x: list(chain(*list(x))))
    
    # Constrain to a single rec_id because, apart from the activity (which is dealt with above) the bc_outputs have the same info for every scanfield.
    bc_output = bc_output[bc_output["rec_id"] == rec_ids[0]]
    bc_output["activity"] = list(bc_output_concatenated.to_numpy())

    # Join stimulus dfs.
    stimuli = stimuli_meta.join(bc_output.set_index("bc_id"), on="bc_id", how="left", rsuffix="_bc")
    stimuli = stimuli.drop(columns="cell_id_bc")
    
    # Join recording dfs.
    labels_df["unique_id"] = labels_df["rec_id"] * 100 + labels_df["roi_id"]
    recording_meta["unique_id"] = recording_meta["rec_id"] * 100 + recording_meta["roi_id"]
    recordings = recording_meta.join(labels_df.set_index("unique_id"), on="unique_id", how="left", rsuffix="_ca")
    recordings = recordings.drop(columns=["cell_id_ca", "rec_id_ca"])

    return stimuli, recordings, setup, noise_full


def _average_calcium_in_identical_comps(rec_df, num_datapoints_per_scanfield):
    num_datapoints = num_datapoints_per_scanfield
    rec_df[[f"ca{i}" for i in range(num_datapoints)]] = pd.DataFrame(rec_df.ca.tolist(), index=rec_df.index)
    rec_df = rec_df.drop(columns="ca")
    mean_df = rec_df.groupby(["branch_ind", "comp_discrete"]).mean()

    # Merge columns into a list of a single column.
    mean_df["ca"]= mean_df[[f"ca{i}" for i in range(num_datapoints)]].values.tolist()
    for i in range(num_datapoints):
        mean_df = mean_df.drop(columns=f"ca{i}")
    return mean_df


def build_avg_recordings(recordings, rec_ids, nseg, num_datapoints_per_scanfield):
    avg_recordings = []
    for rec_id in rec_ids:
        rec_in_scanfield = recordings[recordings["rec_id"] == rec_id].copy()
        
        rec_in_scanfield["comp_discrete"] = np.clip(np.floor(rec_in_scanfield["comp"] * nseg).tolist(), a_min=0, a_max=nseg-1)
        rec_in_scanfield = rec_in_scanfield.drop(columns="cell_id")
        
        avg_recordings_in_scanfield = _average_calcium_in_identical_comps(
            rec_in_scanfield, num_datapoints_per_scanfield
        )
        
        # Make `branch_ind` and `discrete_comp` columns again.
        avg_recordings_in_scanfield = avg_recordings_in_scanfield.reset_index()
    
        avg_recordings.append(avg_recordings_in_scanfield)
    
    avg_recordings = pd.concat(avg_recordings)
    avg_recordings["rec_id"] = avg_recordings["rec_id"].astype(int)
    return avg_recordings


def build_training_data(
    i_amp,
    stimuli,
    avg_recordings,
    rec_ids,
    num_datapoints_per_scanfield,
    number_of_recordings_each_scanfield,
):
    number_of_recordings = len(avg_recordings)
    
    # The currents that will be used as step currents.
    bc_activity = np.stack(stimuli["activity"].to_numpy()).T
    currents = i_amp * np.asarray(bc_activity) / stimuli["num_synapses_of_bc"].to_numpy()
    
    # Labels will also have to go to a dataloader.
    loss_weights = np.zeros((number_of_recordings, len(rec_ids) * num_datapoints_per_scanfield))
    labels = np.zeros((number_of_recordings, len(rec_ids) * num_datapoints_per_scanfield))
    
    cumsum_rec = np.cumsum([0] + number_of_recordings_each_scanfield)
    for i in range(len(rec_ids)):
        rec_id = rec_ids[i]
        start = cumsum_rec[i]
        end = cumsum_rec[i+1]
    
        # Masks for loss.
        loss_weights[start:end, i*num_datapoints_per_scanfield: (i+1)*num_datapoints_per_scanfield] = 1.0
    
        # Labels.
        recordings_in_this_scanfield = avg_recordings[avg_recordings["rec_id"] == rec_id]
        labels_in_this_scanfield = np.asarray(np.stack(recordings_in_this_scanfield["ca"].to_numpy()).T)
        labels[start:end, i*num_datapoints_per_scanfield: (i+1)*num_datapoints_per_scanfield] = labels_in_this_scanfield.T
        i += 1
    
    loss_weights = np.asarray(loss_weights)
    loss_weights = loss_weights.T  # shape (num_images_per_scanfield, 4).
    
    labels = labels.T  # shape (num_images_per_scanfield, 4).
    labels = np.asarray(labels)

    return currents, labels, loss_weights