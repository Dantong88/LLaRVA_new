#!/bin/bash


# TAR_DIR="/p/work1/jquenum/datasets/llarva/v2/tar_dir"
# UNTAR_DIR="/p/work1/jquenum/datasets/llarva/v2/untar_dir"
# # TAR_DIR="/p/work1/jquenum/datasets/llarva/v2/holder"

# # Loop through each tar file in the current directory
# for file in "$TAR_DIR"/*.tar; do
#     # Check if the file is a regular file
#     if [ -f "$file" ]; then

#         # Create a temporary directory named "name_temp"
#         temp_dir="${file%.tar}_temp"

#         # Create the temporary directory
#         mkdir -p "$temp_dir"

#         # Extract the tar file into the temporary directory
#         tar -xf "$file" -C "$temp_dir"
#         echo "Extracted $file to $temp_dir"

#         # Check if the content starts with the word "scratch"
#         if [ -d "$temp_dir/scratch" ]; then

#             cp -r "$temp_dir/scratch/partial_datasets/llarva/rtx/v2/images/"* "$UNTAR_DIR/"
#         else
#             cp -r "$temp_dir"/* "$UNTAR_DIR/"

#         fi

#         echo "Copied contents of $temp_dir to $UNTAR_DIR"

#         # Cleanup: Remove the temporary directory
#         rm -r "$temp_dir"
#         echo "Removed temporary directory $temp_dir"

#     fi
# done

    # local UNTAR_DIR="/p/work1/jquenum/datasets/llarva/v2/untar_dir_fresh"
    # local UNTAR_DIR="/p/work1/jquenum/datasets/llarva/v2/untar_dir"
    # local untar_in_destination="$IM_DIR/untar_dir"

function untar_data() {
    local dataset_name="${1}"

    local TAR_DIR="/p/work1/jquenum/datasets/llarva/v2/tar_dir"
    local DES_IM_DIR="/p/work1/jquenum/datasets/llarva/v2/images"


    # Set the path to the tar file
    local file="${TAR_DIR}/${dataset_name}.tar"
    local destination="$DES_IM_DIR/$dataset_name"

    # Check if the dataset already exists at the destination
    if [ -d "$destination" ]; then
        echo "Dataset $dataset_name already exists at $destination. Deleting existing dataset..."
        rm -rf "$destination"
    fi

    if [ -f "$file" ]; then
        # Create a temporary directory named "name_temp"
        local temp_dir="${file%.tar}_temp"

        # Create the temporary directory
        mkdir -p "$temp_dir"

        # Extract the tar file into the temporary directory
        tar -xf "$file" -C "$temp_dir"
        echo "Extracted $file to $temp_dir"


        # Check if the content of temp_dir is a folder named "scratch"
        if [ -d "$temp_dir/scratch" ]; then
            mv "$temp_dir/scratch/partial_datasets/llarva/rtx/v2/images/"* "$DES_IM_DIR/"
        else
            mv "$temp_dir"/* "$DES_IM_DIR/"
        fi

        echo "Moved $temp_dir to $DES_IM_DIR"

        # Cleanup: Remove the temporary directory
        # rm -r "$temp_dir"
        # echo "Removed temporary directory $temp_dir"
    fi
}



# untar_data "add"
# untar_data "grate"
# untar_data "adjust"
# untar_data "apply"
# untar_data "attach"
# untar_data "austin_buds_dataset_converted_externally_to_rlds"
# untar_data "austin_sailor_dataset_converted_externally_to_rlds"
# untar_data "austin_sirius_dataset_converted_externally_to_rlds"
# untar_data "bake"
# untar_data "bc_z"
# untar_data "berkeley_autolab_ur5"
# untar_data "berkeley_cable_routing"
# untar_data "berkeley_fanuc_manipulation"
# untar_data "berkeley_rpt_converted_externally_to_rlds"
# untar_data "berkeley_rpt_converted_externally_to_rlds_new"
##################
# untar_data "break"
# untar_data "bridge"
# untar_data "brush"
# untar_data "carry"
# untar_data "check"
# untar_data "choose"
# untar_data "close"
# untar_data "cmu_play_fusion"
# untar_data "coat"
# untar_data "columbia_cairlab_pusht_real"
# untar_data "cook"
# untar_data "crush"
# untar_data "cut"
# untar_data "divide"
# untar_data "dlr_edan_shared_control_converted_externally_to_rlds"
# untar_data "dlr_sara_grid_clamp_converted_externally_to_rlds"
# untar_data "dlr_sara_pour_converted_externally_to_rlds"
# untar_data "drink"
# untar_data "drop"
# untar_data "dry"
# untar_data "eat"
# untar_data "empty"
# untar_data "feel"
# untar_data "fill"
# untar_data "filter"
# untar_data "finish"
# untar_data "flatten"
# untar_data "flip"
# untar_data "fold"
# untar_data "form"
# untar_data "fractal20220817_data"
# untar_data "furniture_bench_dataset_converted_externally_to_rlds"
# untar_data "gather"
# untar_data "hang"
# untar_data "hold"
# untar_data "increase"
# untar_data "insert"
# untar_data "jaco_play"
# untar_data "kaist_nonprehensile_converted_externally_to_rlds"
###########################################
# untar_data "knead"
# untar_data "kuka"
# untar_data "language_table"
# untar_data "let-go"
# untar_data "lift"
# untar_data "lock"
# untar_data "look"
# untar_data "lower"
# untar_data "maniskill_dataset_converted_externally_to_rlds"
# untar_data "mark"
# untar_data "measure"
# untar_data "mix"
# untar_data "move"
# untar_data "nyu_franka_play_dataset_converted_externally_to_rlds"
# untar_data "open"
# untar_data "pat"
# untar_data "peel"
# untar_data "pour"
# untar_data "prepare"
# untar_data "press"
# untar_data "pull"
# untar_data "put"
# untar_data "remove"
# untar_data "rip"
# untar_data "robo_net"
# untar_data "roboturk"
# untar_data "roll"
# untar_data "rub"
# untar_data "scoop"
# untar_data "scrape"
# untar_data "screw"
# untar_data "scrub"
# untar_data "search"
#######################################
untar_data "season"
untar_data "serve"
untar_data "set"
untar_data "shake"
untar_data "sharpen"
untar_data "slide"
untar_data "smell"
untar_data "soak"
untar_data "sort"
untar_data "spray"
untar_data "sprinkle"
untar_data "squeeze"
untar_data "stab"
untar_data "stanford_hydra_dataset_converted_externally_to_rlds"
untar_data "stanford_kuka_multimodal_dataset_converted_externally_to_rlds"
untar_data "stanford_mask_vit_converted_externally_to_rlds"
untar_data "stanford_robocook_converted_externally_to_rlds"
untar_data "stretch"
untar_data "switch"
untar_data "taco_play"
untar_data "take"
untar_data "throw"
untar_data "tokyo_u_lsmo_converted_externally_to_rlds"
untar_data "toto"
untar_data "transition"
untar_data "turn"
untar_data "turn-down"
untar_data "turn-off"
untar_data "turn-on"
untar_data "ucsd_pick_and_place_dataset_converted_externally_to_rlds"
untar_data "uncover"
untar_data "unfreeze"
untar_data "unlock"
untar_data "unroll"
untar_data "unscrew"
untar_data "unwrap"
untar_data "use"
untar_data "utaustin_mutex"
untar_data "utokyo_pr2_opening_fridge_converted_externally_to_rlds"
untar_data "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds"
untar_data "utokyo_xarm_pick_and_place_converted_externally_to_rlds"
untar_data "viola"
untar_data "wait"
untar_data "wash"
untar_data "water"
untar_data "wear"


echo "!!!!!!!!! DONE !!!!!!!!!"