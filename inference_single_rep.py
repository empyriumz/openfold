import argparse
import logging
import numpy as np
import os
import torch
import json
import time
from pathlib import Path
from ml_collections import config_dict
from openfold.utils.import_weights import (
    import_jax_weights_,
)
from openfold.model.model import AlphaFold
from openfold.data.parse_fasta_files import process_fasta
from timeit import default_timer as timer
from scripts.utils import logging_related

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)
from openfold.config import model_config
from openfold.data import feature_pipeline, data_pipeline
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)


def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    output_dir,
):
    tmp_fasta_path = os.path.join(output_dir, f"tmp_{os.getpid()}.fasta")
    if len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path, alignment_dir=local_alignment_dir
        )
    else:
        with open(tmp_fasta_path, "w") as fp:
            fp.write("\n".join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path,
            super_alignment_dir=alignment_dir,
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict


def main(conf):
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    model_name = conf.model.alphafold_model
    evoformer_config = model_config(model_name, train=False, low_prec=True)
    model = AlphaFold(evoformer_config).to(device)
    model = model.eval()
    npz_path = os.path.join(conf.model.jax_param_path, "params_" + model_name + ".npz")
    import_jax_weights_(model, npz_path, version=model_name)

    template_featurizer = None
    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    feature_processor = feature_pipeline.FeaturePipeline(evoformer_config.data)
    alignment_dir = conf.data.precomputed_alignments_path
    ID_list, seq_list = process_fasta(conf.data.fasta_path)
    sorted_targets = list(zip(*(ID_list, seq_list)))
    feature_dicts = {}
    with torch.no_grad():
        for tag, seq in sorted_targets:
            feature_dict = feature_dicts.get(tag, None)
            logging.info("Running inference for {}".format(tag))
            t = time.perf_counter()

            if feature_dict is None:
                feature_dict = generate_feature_dict(
                    [tag],
                    [seq],
                    alignment_dir,
                    data_processor,
                    conf.output_path,
                )
                feature_dicts[tag] = feature_dict

            processed_feature_dict = feature_processor.process_features(
                feature_dict,
                mode="predict",
            )

            processed_feature_dict = {
                k: torch.as_tensor(v, device=device)
                for k, v in processed_feature_dict.items()
            }
            out = model(processed_feature_dict)
            out = tensor_tree_map(lambda x: np.array(x.detach().cpu()), out)
            inference_time = time.perf_counter() - t
            logging.info("Inference time: {:.1f}".format(inference_time))
            if conf.data.save_output:
                np.save(conf.output_path + "/" + tag, out["single"])


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Name of configuration file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        conf = json.load(f)

    output_path = None
    if not conf["general"]["debug"]:
        output_path = Path("./evoformer_embedding/") / Path(conf["data"]["ligand"])
        output_path.mkdir(parents=True, exist_ok=True)
        conf["output_path"] = "./" + str(output_path)
        with open(str(output_path) + "/config.json", "w") as f:
            json.dump(conf, f, indent=4)

    conf = config_dict.ConfigDict(conf)
    logging_related(output_path=output_path, training=False, debug=conf.general.debug)
    main(conf)
    end = timer()
    logging.info("Total time used {:.2f}".format(end - start))
