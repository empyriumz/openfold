# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import math
import numpy as np
import os

from openfold.utils.script_utils import (
    parse_fasta,
    run_model,
    prep_output,
    relax_protein,
)
from openfold.utils.import_weights import (
    import_jax_weights_,
)
from openfold.model.model import AlphaFold
logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import pickle
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import random
import time
import torch

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)
from scripts.utils import add_data_args


TRACING_INTERVAL = 50


def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        if args.use_precomputed_alignments is None and not os.path.isdir(
            local_alignment_dir
        ):
            logger.info(f"Generating alignments for {tag}...")

            os.makedirs(local_alignment_dir)

            alignment_runner = data_pipeline.AlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hhsearch_binary_path=args.hhsearch_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                pdb70_database_path=args.pdb70_database_path,
                no_cpus=args.cpus,
            )
            alignment_runner.run(tmp_fasta_path, local_alignment_dir)
        else:
            logger.info(f"Using precomputed alignments for {tag} at {alignment_dir}...")

        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)


def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def run_model(model, batch, tag, args):
    with torch.no_grad():
        # Temporarily disable templates if there aren't any in the batch
        template_enabled = model.config.template.enabled
        model.config.template.enabled = template_enabled and any(
            ["template_" in k for k in batch]
        )

        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        logger.info("Inference time: {:.1f}".format(inference_time))

        model.config.template.enabled = template_enabled

    return out


def prep_output(out, batch, feature_dict, feature_processor, args):
    plddt = out["plddt"]
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    if args.subtract_plddt:
        plddt_b_factors = 100 - plddt_b_factors

    # Prep protein metadata
    template_domain_names = []
    template_chain_index = None
    if (
        feature_processor.config.common.use_templates
        and "template_domain_names" in feature_dict
    ):
        template_domain_names = [
            t.decode("utf-8") for t in feature_dict["template_domain_names"]
        ]

        # This works because templates are not shuffled during inference
        template_domain_names = template_domain_names[
            : feature_processor.config.predict.max_templates
        ]

        if "template_chain_index" in feature_dict:
            template_chain_index = feature_dict["template_chain_index"]
            template_chain_index = template_chain_index[
                : feature_processor.config.predict.max_templates
            ]

    no_recycling = feature_processor.config.common.max_recycling_iters
    remark = ", ".join(
        [
            f"no_recycling={no_recycling}",
            f"max_templates={feature_processor.config.predict.max_templates}",
            f"config_preset={args.config_preset}",
        ]
    )

    # For multi-chain FASTAs
    ri = feature_dict["residue_index"]
    chain_index = (ri - np.arange(ri.shape[0])) / args.multimer_ri_gap
    chain_index = chain_index.astype(np.int64)
    cur_chain = 0
    prev_chain_max = 0
    for i, c in enumerate(chain_index):
        if c != cur_chain:
            cur_chain = c
            prev_chain_max = i + cur_chain * args.multimer_ri_gap

        batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        chain_index=chain_index,
        remark=remark,
        parents=template_domain_names,
        parents_chain_index=template_chain_index,
    )

    return unrelaxed_protein

def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    args,
):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
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


def get_model_basename(model_path):
    return os.path.splitext(os.path.basename(os.path.normpath(model_path)))[0]


def make_output_directory(output_dir, model_name, multiple_model_mode):
    if multiple_model_mode:
        prediction_dir = os.path.join(output_dir, "predictions", model_name)
    else:
        prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir


def count_models_to_evaluate(openfold_checkpoint_path, jax_param_path):
    model_count = 0
    if openfold_checkpoint_path:
        model_count += len(openfold_checkpoint_path.split(","))
    if jax_param_path:
        model_count += len(jax_param_path.split(","))
    return model_count


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.custom_template is None:
        model_list = ["model_1", "model_2", "model_3", "model_4", "model_5"]
    else:
        model_list = ["model_1", "model_2"]
    best_plddt = 0
    for model_name in model_list:
        config = model_config(model_name)
        model = AlphaFold(config)
        model = model.eval()
        npz_path = os.path.join(
            args.jax_param_path, "params_" + model_name + ".npz"
        )
        import_jax_weights_(model, npz_path, version=model_name)
        model = model.to(args.model_device)
        if args.trace_model:
            if not config.data.predict.fixed_size:
                raise ValueError(
                    "Tracing requires that fixed_size mode be enabled in the config"
                )
        if args.custom_template is None:
            template_featurizer = templates.TemplateHitFeaturizer(
                mmcif_dir=args.template_mmcif_dir,
                max_template_date=args.max_template_date,
                max_hits=config.data.predict.max_templates,
                kalign_binary_path=args.kalign_binary_path,
                release_dates_path=args.release_dates_path,
                obsolete_pdbs_path=args.obsolete_pdbs_path,
            )
        else:
            logger.info("Use custom template")
            template_featurizer = None

        data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )

        output_dir_base = args.output_dir
        random_seed = args.data_random_seed
        if random_seed is None:
            random_seed = random.randrange(2**32)

        np.random.seed(random_seed)
        torch.manual_seed(random_seed + 1)

        feature_processor = feature_pipeline.FeaturePipeline(config.data)
        if not os.path.exists(output_dir_base):
            os.makedirs(output_dir_base)
        if args.use_precomputed_alignments is None:
            alignment_dir = os.path.join(output_dir_base, "alignments")
        else:
            alignment_dir = args.use_precomputed_alignments

        tag_list = []
        seq_list = []
        for fasta_file in list_files_with_extensions(args.fasta_dir, (".fasta", ".fa")):
            # Gather input sequences
            with open(os.path.join(args.fasta_dir, fasta_file), "r") as fp:
                data = fp.read()

            tags, seqs = parse_fasta(data)
            # assert len(tags) == len(set(tags)), "All FASTA tags must be unique"
            tag = "-".join(tags)

            tag_list.append((tag, tags))
            seq_list.append(seqs)

        seq_sort_fn = lambda target: sum([len(s) for s in target[1]])
        sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)
        feature_dicts = {}
        cur_tracing_interval = 0
        for (tag, tags), seqs in sorted_targets:
            # Does nothing if the alignments have already been computed
            precompute_alignments(tags, seqs, alignment_dir, args)

            feature_dict = feature_dicts.get(tag, None)
            if feature_dict is None:
                feature_dict = generate_feature_dict(
                    tags,
                    seqs,
                    alignment_dir,
                    data_processor,
                    args,
                )
            if args.custom_template is not None:
                feature_dict = templates.single_template_process(
                    feature_dict, args.custom_template
                )

                if args.trace_model:
                    n = feature_dict["aatype"].shape[-2]
                    rounded_seqlen = round_up_seqlen(n)
                    feature_dict = pad_feature_dict_seq(
                        feature_dict,
                        rounded_seqlen,
                    )

                feature_dicts[tag] = feature_dict

            processed_feature_dict = feature_processor.process_features(
                feature_dict,
                mode="predict",
            )

            processed_feature_dict = {
                k: torch.as_tensor(v, device=args.model_device)
                for k, v in processed_feature_dict.items()
            }

            if args.trace_model:
                if rounded_seqlen > cur_tracing_interval:
                    logger.info(f"Tracing model at {rounded_seqlen} residues...")
                    t = time.perf_counter()
                    trace_model_(model, processed_feature_dict)
                    tracing_time = time.perf_counter() - t
                    logger.info(f"Tracing time: {tracing_time}")
                    cur_tracing_interval = rounded_seqlen

            out = run_model(model, processed_feature_dict, tag, args.output_dir)

            # Toss out the recycling dimensions --- we don't need them anymore
            processed_feature_dict = tensor_tree_map(
                lambda x: np.array(x[..., -1].cpu()), processed_feature_dict
            )
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
            mean_plddt = np.mean(out["plddt"])
            logger.info("Mean plddt for {}: {:.2f}".format(model_name, mean_plddt))
            if mean_plddt > best_plddt:
                best_plddt = mean_plddt
                best_protein = prep_output(
                    out, processed_feature_dict, feature_dict, feature_processor, args
                )
                output_name = "{}_{}_{:.2f}".format(tag, model_name, best_plddt)
                if args.output_postfix is not None:
                    output_name = f"{output_name}_{args.output_postfix}"

    if not args.skip_relaxation:
        relax_protein(
                    config,
                    args.model_device,
                    best_protein,
                    args.output_dir,
                    output_name,
                )
        
    if args.save_outputs:
        output_dict_path = os.path.join(
            args.output_dir, f"{output_name}_output_dict.pkl"
        )
        with open(output_dict_path, "wb") as fp:
            pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Model output written to {output_dict_path}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_dir",
        type=str,
        help="Path to directory containing FASTA files, one sequence per file",
    )
    parser.add_argument(
        "template_mmcif_dir",
        type=str,
    )
    parser.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored.""",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--model_device",
        type=str,
        default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")""",
    )
    parser.add_argument(
        "--config_preset",
        type=str,
        default="model_1",
        help="""Name of a model config preset defined in openfold/config.py""",
    )
    parser.add_argument(
        "--jax_param_path",
        type=str,
        default="/hpcgpfs01/scratch/xdai/openfold/params",
        help="""Path to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params""",
    )
    parser.add_argument(
        "--openfold_checkpoint_path",
        type=str,
        default=None,
        help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file""",
    )
    parser.add_argument(
        "--custom_template",
        type=str,
        default=None,
        help="""Using custom template for the inference.""",
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        default=False,
        help="Whether to save all model outputs, including embeddings, etc.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="""Number of CPUs with which to run alignment tools""",
    )
    parser.add_argument(
        "--preset", type=str, default="full_dbs", choices=("reduced_dbs", "full_dbs")
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default=None,
        help="""Postfix for output prediction filenames""",
    )
    parser.add_argument("--data_random_seed", type=str, default=None)
    parser.add_argument(
        "--skip_relaxation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--multimer_ri_gap",
        type=int,
        default=200,
        help="""Residue index offset between multiple sequences, if provided""",
    )
    parser.add_argument(
        "--trace_model",
        action="store_true",
        default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs.""",
    )
    parser.add_argument(
        "--subtract_plddt",
        action="store_true",
        default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself""",
    )
    add_data_args(parser)
    args = parser.parse_args()

    if args.model_device == "cpu" and torch.cuda.is_available():
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )
    main(args)
