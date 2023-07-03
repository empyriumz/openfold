from Bio import SeqIO


def process_fasta(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    seq_list, ID_list = [], []
    with open(
        "/sdcc/u/xdai/Protein-ligand-binding/data_processing/mismatch_ids.txt"
    ) as file:
        wrong_ids = set(line.strip() for line in file)

    for rec in records:
        if rec.id.upper() in wrong_ids:
            continue
        else:
            seq_list.append(str(rec.seq))
            ID_list.append(rec.id)

    assert len(ID_list) == len(seq_list), "broken fasta input"
    assert len(seq_list) == len(set(seq_list)), "duplicate entries found"
    del records
    return ID_list, seq_list
