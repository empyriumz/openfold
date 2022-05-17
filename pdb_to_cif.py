import os
from pathlib import Path

def run_shell(text, print_text=True):
    """Utility to run a string as a shell script and toss output"""

    if print_text:
        print("RUNNING:", text)
    result = os.system(text)
    return result

def run_pdb_to_cif(pdb_file, content_dir="/host/openfold"):
    assert content_dir is not None
    if hasattr(pdb_file, "as_posix"):
        pdb_file = pdb_file.as_posix()  # make it a string
    output_file = pdb_file.replace(".pdb", ".cif")

    p = os.path.join(content_dir, "maxit-v11.100-prod-src")
    b = os.path.join(p, "bin", "process_entry")
    os.environ["RCSBROOT"] = p
    run_shell(
        "%s -input %s -input_format pdb -output %s -output_format cif"
        % (b, pdb_file, output_file)
    )
    return Path(output_file)

run_pdb_to_cif("/host/openfold/fasta_data/7lx5/7lx5_23566_one_chain_deposit.pdb")