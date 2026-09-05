import os
import numpy as np
import pandas as pd
import MDAnalysis as mda

from MDAnalysis.lib.distances import capped_distance


# =========================================================
# USER SETTINGS
# =========================================================

# Input trajectory in multi-model PDB format
trajectory_pdb = "basal_arr2.pdb"

# Output directory
output_directory = "outputs"

# Reference region used for residue-level contact calculation
reference_start = 1
reference_end = 353

# Contact cutoff in Angstrom
# 0.6 nm = 6 Angstrom
cutoff = 6.0


# =========================================================
# C-TERMINAL REGIONS
# =========================================================

regions = {
    "proximal": (354, 383),
    "distal1": (393, 401),
    "distal2": (402, 409),
    "distal3": (410, 418),
}


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(output_directory, exist_ok=True)


# =========================================================
# CHECK INPUT FILE
# =========================================================

if not os.path.exists(trajectory_pdb):
    raise FileNotFoundError(
        f"Input trajectory not found:\n{trajectory_pdb}"
    )

print("Input trajectory:")
print(trajectory_pdb)


# =========================================================
# LOAD TRAJECTORY
# =========================================================

u = mda.Universe(trajectory_pdb)

print("\n" + "=" * 70)
print("TRAJECTORY INFORMATION")
print("=" * 70)

print("Number of frames   :", len(u.trajectory))
print("Number of atoms    :", u.atoms.n_atoms)
print("Number of residues :", len(u.residues))


# =========================================================
# SELECT REFERENCE REGION
# =========================================================

reference = u.select_atoms(
    f"protein and resid {reference_start}:{reference_end}"
)

if reference.n_atoms == 0:
    raise ValueError(
        f"No atoms found in reference region "
        f"{reference_start}-{reference_end}."
    )

reference_residues = reference.residues
n_reference_residues = len(reference_residues)

print("\nReference region:")
print(f"{reference_start}-{reference_end}")
print("Number of residues :", n_reference_residues)
print("Number of atoms    :", reference.n_atoms)


# =========================================================
# MAP REFERENCE ATOMS TO RESIDUE INDICES
# =========================================================

atom_to_reference_residue = {}

for residue_index, residue in enumerate(reference_residues):
    for atom in residue.atoms:
        atom_to_reference_residue[atom.index] = residue_index


# =========================================================
# WRITE FIRST FRAME TEMPORARILY
# =========================================================

temp_firstframe_pdb = os.path.join(
    output_directory,
    "firstframe_temp.pdb"
)

u.trajectory[0]

with mda.Writer(
    temp_firstframe_pdb,
    n_atoms=u.atoms.n_atoms,
    multiframe=False
) as writer:
    writer.write(u.atoms)


# =========================================================
# ANALYZE EACH C-TERMINAL REGION
# =========================================================

all_results = {}

for region_name, (target_start, target_end) in regions.items():

    print("\n\n")
    print("#" * 70)
    print(
        f"ANALYZING {region_name.upper()} REGION "
        f"({target_start}-{target_end})"
    )
    print("#" * 70)

    # -----------------------------------------------------
    # Select target C-terminal region
    # -----------------------------------------------------

    target = u.select_atoms(
        f"protein and resid {target_start}:{target_end}"
    )

    if target.n_atoms == 0:
        print(
            f"WARNING: No atoms found in region "
            f"{target_start}-{target_end}."
        )
        continue

    print("\nTarget region:")
    print(f"{region_name} = {target_start}-{target_end}")
    print("Number of residues :", len(target.residues))
    print("Number of atoms    :", target.n_atoms)

    # -----------------------------------------------------
    # Contact analysis
    # -----------------------------------------------------

    all_frame_contacts = []
    n_frames = len(u.trajectory)

    for frame_index, ts in enumerate(u.trajectory):

        pairs = capped_distance(
            target.positions,
            reference.positions,
            max_cutoff=cutoff,
            box=ts.dimensions,
            return_distances=False
        )

        frame_contacts = np.zeros(
            n_reference_residues,
            dtype=int
        )

        for target_local_index, reference_local_index in pairs:

            reference_atom = reference[
                reference_local_index
            ]

            residue_index = atom_to_reference_residue[
                reference_atom.index
            ]

            frame_contacts[residue_index] += 1

        all_frame_contacts.append(frame_contacts)

        if (
            frame_index == 0
            or (frame_index + 1) % 100 == 0
            or (frame_index + 1) == n_frames
        ):
            print(
                f"{region_name}: Frame "
                f"{frame_index + 1}/{n_frames} processed"
            )

    all_frame_contacts = np.asarray(
        all_frame_contacts
    )

    # -----------------------------------------------------
    # Per-residue statistics
    # -----------------------------------------------------

    mean_contacts = np.mean(
        all_frame_contacts,
        axis=0
    )

    std_contacts = np.std(
        all_frame_contacts,
        axis=0
    )

    min_contacts_frame = np.min(
        all_frame_contacts,
        axis=0
    )

    max_contacts_frame = np.max(
        all_frame_contacts,
        axis=0
    )

    # -----------------------------------------------------
    # Min-max normalization to 0-1
    #
    # Each C-terminal region is normalized independently.
    # -----------------------------------------------------

    contact_min = mean_contacts.min()
    contact_max = mean_contacts.max()

    if contact_max > contact_min:

        normalized_contacts = (
            (mean_contacts - contact_min)
            /
            (contact_max - contact_min)
        )

    else:

        normalized_contacts = np.zeros_like(
            mean_contacts
        )

    print("\nNORMALIZATION")
    print("Minimum mean contact :", contact_min)
    print("Maximum mean contact :", contact_max)

    print(
        "Normalized range     :",
        normalized_contacts.min(),
        "-",
        normalized_contacts.max()
    )

    # -----------------------------------------------------
    # Residue ID -> normalized B-factor value
    # -----------------------------------------------------

    beta_by_resid = {
        int(residue.resid): float(beta)
        for residue, beta in zip(
            reference_residues,
            normalized_contacts
        )
    }

    # -----------------------------------------------------
    # Create result table
    # -----------------------------------------------------

    contact_column_name = (
        f"Mean_atom_contacts_with_"
        f"{target_start}_{target_end}"
    )

    results = pd.DataFrame({
        "Resid": [
            residue.resid
            for residue in reference_residues
        ],

        "Resname": [
            residue.resname
            for residue in reference_residues
        ],

        contact_column_name:
            mean_contacts,

        "Normalized_contact_0_1":
            normalized_contacts,

        "Std_atom_contacts":
            std_contacts,

        "Min_atom_contacts":
            min_contacts_frame,

        "Max_atom_contacts":
            max_contacts_frame
    })

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    output_csv = os.path.join(
        output_directory,
        (
            f"arr2_contacts_"
            f"{reference_start}_{reference_end}_with_"
            f"{target_start}_{target_end}_"
            f"{region_name}.csv"
        )
    )

    results.to_csv(
        output_csv,
        index=False
    )

    # -----------------------------------------------------
    # Save PDB with normalized values in B-factor column
    # -----------------------------------------------------

    output_pdb = os.path.join(
        output_directory,
        (
            f"arr2_ctail_contacts_"
            f"{region_name}_beta.pdb"
        )
    )

    colored_atoms = 0

    with open(
        temp_firstframe_pdb,
        "r"
    ) as inp, open(
        output_pdb,
        "w"
    ) as out:

        for line in inp:

            if line.startswith(
                ("ATOM  ", "HETATM")
            ):

                resid_text = line[22:26].strip()

                try:
                    resid = int(resid_text)

                except ValueError:
                    out.write(line)
                    continue

                if resid in beta_by_resid:

                    beta = beta_by_resid[resid]
                    colored_atoms += 1

                else:

                    beta = 0.0

                line_no_newline = line.rstrip("\n")

                if len(line_no_newline) < 66:
                    line_no_newline = (
                        line_no_newline.ljust(66)
                    )

                new_line = (
                    line_no_newline[:60]
                    + f"{beta:6.2f}"
                    + line_no_newline[66:]
                    + "\n"
                )

                out.write(new_line)

            else:

                out.write(line)

    # -----------------------------------------------------
    # Print top 20 contacting residues
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"TOP CONTACTING RESIDUES WITH "
        f"{region_name.upper()} "
        f"({target_start}-{target_end})"
    )

    print("=" * 70)

    top20 = (
        results
        .sort_values(
            contact_column_name,
            ascending=False
        )
        .head(20)
    )

    print(
        top20[
            [
                "Resid",
                "Resname",
                contact_column_name,
                "Normalized_contact_0_1"
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # Store results in memory
    # -----------------------------------------------------

    all_results[region_name] = results

    print("\nCSV output:")
    print(output_csv)

    print("\nPDB output:")
    print(output_pdb)

    print(
        "\nAtoms assigned normalized B-factor values:",
        colored_atoms
    )


# =========================================================
# REMOVE TEMPORARY FILE
# =========================================================

if os.path.exists(temp_firstframe_pdb):
    os.remove(temp_firstframe_pdb)


# =========================================================
# FINISHED
# =========================================================

print("\n\n")
print("=" * 70)
print("ALL ANALYSES COMPLETED")
print("=" * 70)

print("\nAnalyzed C-terminal regions:")

for region_name, (
    target_start,
    target_end
) in regions.items():

    print(
        f"{region_name:10s} : "
        f"{target_start}-{target_end}"
    )
